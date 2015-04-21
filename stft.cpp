//STFT implementation for audio processing
//

#include <exception>
#include <mad.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <ffts/ffts.h>
#include <iostream>
#include <math.h>

using namespace std;
using namespace Eigen;

#ifdef SHRT_MAX
#undef SHRT_MAX
#endif

#define SHRT_MAX (32767)
#define INPUT_BUFFER_SIZE	(5*8192)
#define OUTPUT_BUFFER_SIZE	8192 // Must be an integer multiple of 4

#define PI 3.14159265358979323846264334
#define UNIT_TEST_HANDLE 1000
#define DEFAULT_f 1024

//A few Eigen typedefs
typedef float floatnum;
//Complex number matrix
typedef Matrix<complex<floatnum>, Dynamic, Dynamic> MatrixS;
//Real number matrix
typedef Matrix<floatnum, Dynamic, Dynamic> MatrixSr;
//Complex number vector
typedef Matrix<complex<floatnum>, Dynamic, 1> VectorS;
//Real number vector
typedef Matrix<floatnum, Dynamic, 1> VectorSr;

//Build STFT Matrix 
template <typename T> int buildMatrix (MatrixS& spectrum, T& reader, int f, int w, int h);
//Build vector with audio signal (from STFT matrix)
int buildWave (MatrixS &spec,VectorS &wave, int w, int h);
//Return hann window (or rectanular if w=0)
template <typename T> int hannWindow (T& window, int f, int w);
//Do fft
int ffts (floatnum *input, floatnum *output, size_t size);
//Do ifft
int iffts (floatnum *input, floatnum *output, size_t size);
//Frequency domain processing
int processMatrix (MatrixS& spec, MatrixSr& mask_error, int partial, int power, floatnum gain);
//Save audio signal in wav file
int vectorToWav (VectorS& v, const std::string& file, int sr);
//Transform to time domain and save in wav file (minimizing memory)
int STFTToWav (MatrixS &spec, int w, int h, const std::string& file, int sr, floatnum max);

//Exceptions

//Cannot open file
class BadFile : public std::exception {
public:
    const char* what() const throw ()
    {
        return str;
    }
private:
    static const char str[];
};

const char BadFile::str[] = "Cannot open file";

//Wrong file format
class BadFormat : public std::exception {
public:
    const char* what() const throw ()
    {
        return str;
    }
private:
    static const char str[];
};

const char BadFormat::str[] = "Wrong file format";

//Wav file reader class
class WavFileReader {
public:
    WavFileReader (const std::string& fname);
    ~WavFileReader ();
    template <typename T> int readSamples(T& v, int size, int h);
    int get_sample_rate () {
        return sample_rate;
    }
private:
    char id[5];
    uint32_t size;
    int16_t format_tag, channels, block_align, bits_per_sample;
    uint32_t format_length, sample_rate, avg_bytes_sec, data_size;
    FILE *fp;
};

//Unit test reader class
class UnitTestReader 
{
public:
    UnitTestReader (VectorSr _v);
    int readSamples (VectorS& _v, int size, int h); 

private:
    VectorSr vec;
    int i;
};

//MP3 file reader class
class MP3FileReader
{
public:
    MP3FileReader (const std::string& _file);
    ~MP3FileReader ();
    int readNextFrame();
    int readSamples (VectorS &v, int size, int h);
    int get_sample_rate () 
    {
        return samplerate;
    }

private:    
    int size;
	FILE* file;
	mad_stream stream;
	mad_frame frame;
	mad_synth synth;
	mad_timer_t timer;
    int samplerate;
    int leftSamples;
	int offset;
	unsigned char inputBuffer[INPUT_BUFFER_SIZE];
};

UnitTestReader::UnitTestReader (VectorSr v) : i(0)
{
    vec=v;
}

int UnitTestReader::readSamples (VectorS& v, int size, int h)
{
	int idx = 0;

    if (h>=size)
        h-=size; //number of elements to ignore
    else {
        int j=h;
        for (idx=0; idx<size-h; ++idx, ++j)
            v(idx)=v(j);
        h=0;
    }

    while (idx!=size) {
        if (i>=vec.size())
            return 0;
        if (h)
            --h;
        else {
            v(idx)=vec(i);
            idx++;
        }
        i++;
    }
    return size;
}

WavFileReader::WavFileReader (const std::string& fname) 
{
    fp = fopen(fname.c_str(),"rb");
    if (fp) {
        fread(id, sizeof(char), 4, fp);
        id[4] = '\0';
        
        if (!strcmp(id, "RIFF")) {
            fread(&size, sizeof(uint32_t), 1, fp);
            fread(id, sizeof(char), 4, fp);
            id[4] = '\0';
            
            if (!strcmp(id,"WAVE")) {
                fread(id, sizeof(char), 4, fp);
                fread(&format_length, sizeof(uint32_t),1,fp);
                fread(&format_tag, sizeof(int16_t), 1, fp);
                fread(&channels, sizeof(int16_t),1,fp);
                fread(&sample_rate, sizeof(uint32_t), 1, fp);
                fread(&avg_bytes_sec, sizeof(uint32_t), 1, fp);
                fread(&block_align, sizeof(int16_t), 1, fp);
                fread(&bits_per_sample, sizeof(int16_t), 1, fp);
                fread(id, sizeof(char), 4, fp);
                fread(&data_size, sizeof(uint32_t), 1, fp); 
                if (bits_per_sample!=16) {
                    fclose (fp);
                    throw (BadFormat());
                }
            }
            else {
                fclose (fp);
                cout<<"Error: RIFF file but not a wave file\n";
                throw (BadFormat());
            }
        }
        else {
            fclose (fp);
            cout<<"Error: not a RIFF file\n";
            throw (BadFormat());
        }
    }
    else 
        throw (BadFile());
}

template <typename T> int WavFileReader::readSamples(T& v, int size, int h)
{
	float sum = 0;
	int idx = 0;
    int16_t sample;
    floatnum value;

    if (h>=size)
        h-=size; //number of elements to ignore
    else {
        int j=h;
        for (idx=0; idx<size-h; ++idx, ++j)
            v(idx)=v(j);
        h=0;
    }

	while (idx != size)
	{
        if (!fread(&sample, sizeof(int16_t), 1, fp))
            return 0;
        else 
            value = (floatnum) sample;

        if (channels==2) {
            if (!fread(&sample, sizeof(int16_t), 1, fp))
                return 0;
            else { 
                value += (floatnum) sample;
                value /=2;
            }
        }
        value/=SHRT_MAX;
        if (h)
            --h;
        else {
            v(idx)=value;
            idx++;
        }	
	}

	return size;
}

WavFileReader::~WavFileReader ()
{
    fclose (fp);
}

MP3FileReader::~MP3FileReader ()
{
    fclose(file);
    mad_synth_finish(&synth);
    mad_frame_finish(&frame);
    mad_stream_finish(&stream);
}

//Class used to write WAV files
class WavWriter {
private:
    std::ofstream stream;
    template <typename T>
        void write(std::ofstream& stream, const T& t) {
            stream.write((const char*)&t, sizeof(T));
        }

        void writeWAVData(size_t buf, size_t samples, int sampleRate, int16_t channels)
        {
            int bytes=samples*channels*buf;

            stream.write("RIFF", 4);
            write<int32_t>(stream, 36 + bytes);
            stream.write("WAVE", 4);
            stream.write("fmt ", 4);
            write<int32_t>(stream, 16);
            write<int16_t>(stream, 1);                         // Format (1 = PCM)
            write<int16_t>(stream, channels);                  // Channels
            write<int32_t>(stream, sampleRate);                  // Sample Rate
            write<int32_t>(stream, sampleRate * channels * buf); // Byterate
            write<int16_t>(stream, channels * buf);            // Frame size
            write<int16_t>(stream, 8 * buf);                   // Bits per sample
            stream.write("data", 4);
            stream.write((const char*)&bytes, 4);
        }
public:
    WavWriter (const char* outFile, size_t buf, 
            size_t samples, int sampleRate, int16_t channels) :
        stream (outFile, std::ios::binary) {
            writeWAVData (buf, samples, sampleRate, channels);
        }

    template <typename SampleType> void writeFrame (SampleType *buf, size_t bufsize) 
    {
        stream.write ((const char*)buf, bufsize);
    }

};

int vectorToWav (VectorS& v, const std::string& file, int sr) 
{
    floatnum max;

    max=v.real().array().abs().maxCoeff();
    v=v*(1/max)*SHRT_MAX;

    WavWriter wv (file.c_str(), sizeof(int16_t), v.rows(), sr, 1);

    for (int i=0; i<v.rows(); ++i) {
        int16_t value=(int16_t)v.real()(i);
        wv.writeFrame (&value, sizeof(int16_t));            
    }

    return 0;
}

MP3FileReader::MP3FileReader (const std::string& _file)
{
	FILE* fileHandle = fopen (_file.c_str(), "rb");

	if (fileHandle == 0)
        throw (BadFile());

    leftSamples = 0;
	file = fileHandle;
	fseek (fileHandle, 0, SEEK_END);
	size = ftell (fileHandle);
	rewind (fileHandle);

	mad_stream_init(&stream);
	mad_frame_init(&frame);
	mad_synth_init(&synth);
	mad_timer_reset(&timer);
}

int MP3FileReader::readNextFrame ()
{
	do
	{
		if (stream.buffer == 0 || stream.error == MAD_ERROR_BUFLEN )
		{
			int inputBufferSize = 0;
			if( stream.next_frame != 0 )
			{
				int leftOver = stream.bufend - stream.next_frame;
				for( int i = 0; i < leftOver; i++ )
					inputBuffer[i] = stream.next_frame[i];
				int readBytes = fread( inputBuffer + leftOver, 1, INPUT_BUFFER_SIZE - leftOver, file );
				if( readBytes == 0 )
					return 0;
				inputBufferSize = leftOver + readBytes;
			}
			else
			{
				int readBytes = fread( inputBuffer, 1, INPUT_BUFFER_SIZE, file );
				if( readBytes == 0 )
					return 0;
				inputBufferSize = readBytes;
			}

			mad_stream_buffer(&stream, inputBuffer, inputBufferSize );
			stream.error = MAD_ERROR_NONE;
		}

		if (mad_frame_decode (&frame, &stream))
		{
            samplerate=frame.header.samplerate;
			if (stream.error == MAD_ERROR_BUFLEN ||(MAD_RECOVERABLE(stream.error)))
				continue;
			else
				return 0;
		}
		else
			break;
	} while (true);

	mad_timer_add (&timer, frame.header.duration);
	mad_synth_frame (&synth, &frame);
	leftSamples = synth.pcm.length;
	offset = 0;

	return -1;
}


int MP3FileReader::readSamples(VectorS& v, int size, int h)
{
	int idx = 0;

    if (h>=size)
        h-=size; //number of elements to ignore
    else {
        int j=h;
        for (idx=0; idx<size-h; ++idx, ++j)
            v(idx)=v(j);
        h=0;
    }

	while (idx != size)
	{
		if(leftSamples > 0 )
		{
			for( ; idx < size && offset < synth.pcm.length; leftSamples--, offset++ )
			{
				floatnum value = (floatnum) mad_f_todouble(synth.pcm.samples[0][offset]);
				
				if( MAD_NCHANNELS(&frame.header) == 2 )
				{
					value += (floatnum) mad_f_todouble(synth.pcm.samples[1][offset]);
					value /= 2;
				}
                if (h)
                    --h;
                else {
    				v(idx)=value;
				    idx++;
                }
			}
		}
		else
		{
			int result = readNextFrame();
			if (result == 0)
				return 0;
		}

	}

	return size;
}

template <typename T> int buildMatrix (MatrixS& spectrum, T& reader, int f, int w, int h)
{
    if (f==0)
        f=DEFAULT_f;
    if (h==0)
        h=f/4;

    spectrum.resize (1+f/2, 1);
    VectorS v(f);
    VectorS vwin(f);
    VectorS t(f);
    VectorS win(f); 

    hannWindow (win, f, w);

    int nsamples=reader.readSamples (v, f, f);

    for (int i=0; nsamples==f; ++i) {
        //Don't modify v vector (original samples are reused in readSamples())
        vwin=v.cwiseProduct(win);
        //This cast is safe according to C++11 (and should work with
        //C++03)
        ffts (reinterpret_cast<floatnum*> (vwin.data()), reinterpret_cast<floatnum*> (t.data()), f);

        if (i==spectrum.cols())
            spectrum.conservativeResize (NoChange, i+1);

        spectrum.col(i)=t.head(1+f/2);
        nsamples=reader.readSamples (v, f, h);
    }
    //Release ffts resources
    ffts (NULL, NULL, 0);

    return 0;
}

int buildWave (MatrixS &spec, VectorS &wave, int w, int h)
{
    int f=2*(spec.rows()-1);
    VectorS win(f);

    if (h==0)
        h=f/4;

    hannWindow (win, f, w);
    //Make stft-istft loop be identity for 25% hop (see matlab reference
    //implementation)
    win=2.0/3.0*win;
    wave = VectorS::Zero(f+(spec.cols()-1)*h);
    VectorS ft (f);
    VectorS t (f);

    for (int i=0; i<spec.cols(); ++i) {
        ft.head(spec.rows())=spec.col(i);
        ft.tail(f-spec.rows())=spec.col(i).segment(1, (f-1)/2).conjugate().reverse();
        //This cast is safe according to C++11 (and should work with
        //C++03)
        iffts (reinterpret_cast<floatnum*> (ft.data()), reinterpret_cast<floatnum*> (t.data()), f);
        wave.segment(i*h, f)+=t.cwiseProduct(win); 
    }

    //release FFTS resources
    iffts (NULL, NULL, 0);

    return 0;
}

int STFTToWav (MatrixS &spec, int w, int h, const std::string& file, int sr, floatnum max)
{
    int f=2*(spec.rows()-1);
    VectorS win(f);
    WavWriter wv (file.c_str(), sizeof(int16_t), f+(spec.cols()-1)*h, sr, 1);
    if (h==0)
        h=f/4;

    hannWindow (win, f, w);
    //Make stft-istft loop be identity for 25% hop (see matlab reference
    //implementation)
    win=2.0/3.0*win;
    VectorS wave = VectorS::Zero(f);
    VectorS ft (f);
    VectorS t (f);

    floatnum scale=(1.0/max)*(floatnum)SHRT_MAX;
    for (int i=0; i<spec.cols(); ++i) {
        ft.head(spec.rows())=spec.col(i);
        ft.tail(f-spec.rows())=spec.col(i).segment(1, (f-1)/2).conjugate().reverse();
        //This cast is safe according to C++11 (and should work with
        //C++03)
        iffts (reinterpret_cast<floatnum*> (ft.data()), reinterpret_cast<floatnum*> (t.data()), f);
        wave.segment(0, f)+=t.cwiseProduct(win); 

        //last column? write full window 
        if (i==spec.cols()-1)
            h=f;

        for (int j=0; j<h && j<f; ++j) {
            int16_t value = (int16_t)(wave.real()(j)*scale);
            wv.writeFrame (&value, sizeof(int16_t));
        }
        
        if (f-h>0) {
            wave.segment (0, f-h) = wave.segment (h, f-h);
            wave.segment (f-h, h) = VectorS::Zero (h);
        }

        else
            wave = VectorS::Zero (f);
    }

    //release FFTS resources
    iffts (NULL, NULL, 0);

    return 0;
}


int ffts (floatnum *input, floatnum *output, size_t size)
{
    static ffts_plan_t *p=NULL; 
    static size_t last_size=0;

    if (last_size!=size) {
        last_size=size;
        if (p!=NULL) {
            ffts_free (p);
            p=NULL;
        }
        if (size!=0) {
            p=ffts_init_1d (size, -1);
        }
        else
            return 0;
    }

    ffts_execute(p, input, output);

    return 0;
}

int iffts (floatnum *input, floatnum *output, size_t size)
{
    static ffts_plan_t *p=NULL; 
    static size_t last_size=0;

    if (last_size!=size) {
        last_size=size;
        if (p!=NULL) {
            ffts_free (p);
            p=NULL;
        }
        if (size!=0) {
            p=ffts_init_1d (size, 1);
        }
        else
            return 0;
    }

    floatnum coef=1/(float)size;

    ffts_execute(p, input, output);
    for (int i=0; i<size; i++) {
        output[i*2]*=coef;
        output[i*2+1]*=coef;
    }

    return 0;
}

template <typename T> int hannWindow (T& window, int f, int w)
{
    int i_f, i_w;

    //special case: rectangular window
    if (w==0) {
        for (i_f=0; i_f<f; i_f++)
            window(i_f)=1.0;
        return 0;
    }

    //w must be odd (see matlab reference implementation)
    if (w%2 == 0)
        ++w;

    window=T::Zero(f);

    if (f>=w) {
        i_f=(f-w+1)/2;
        i_w=0;
    }
    else {
        i_f=0;
        i_w=(w-f)/2;
    }

    for (; i_f<f && i_w<w; ++i_f, ++i_w) 
        window(i_f)=0.5*(1-cos(2*PI*i_w/(w-1)));

    return 0;
}

//Functor to get the phase angle of a complex number
struct angle {
    complex<floatnum> operator()(const complex<floatnum>& num) const 
    {
        return arg(num);
    }
};

//Process spec STFT matrix
int processMatrix (MatrixS& spec, MatrixSr& mask_error, int partial, int power, floatnum gain)
{
    complex<floatnum> i;
    i.real()=0.0;
    i.imag()=1.0;

    MatrixS partial_spectrum=spec.topRows (partial);
    MatrixS remain_spectrum=spec.bottomRows (spec.rows()-partial);
    MatrixS phase=partial_spectrum.unaryExpr(angle());
    MatrixSr enhanced_spectrum=partial_spectrum.array().abs().pow(power).matrix();
    
    spec << ((phase.array()*i).exp().array()*enhanced_spectrum.array()).matrix(), remain_spectrum; 

    MatrixSr threshold=MatrixSr::Constant (enhanced_spectrum.rows(), enhanced_spectrum.cols(), gain);
    MatrixSr m=(enhanced_spectrum.array() > threshold.array()).cast<floatnum>();
    MatrixSr masked_spectrum=(m.array()*enhanced_spectrum.array()).matrix();
    mask_error=enhanced_spectrum-masked_spectrum;

    return 0;
}

int main(int argc, char** argv)
{
    MatrixS spec;
    MatrixSr mask_error;
    VectorS wave;

    if (argc<2) {
        cout << "Performing unit test" << endl;

        //Initialize vector for unit test
        VectorSr vtest (96);
        for (int i=0; i<96; i++)
            vtest(i)=i;

        //Create unit test reader
        UnitTestReader utr(vtest);

        //build STFT matrix. Parameters are:
        //spec = stft return matrix
        //utr = data reader (unit test in this case)
        //f=32, w=0, h=32
        buildMatrix (spec, utr, 32, 0, 32);
        cout << "STFT matrix" << endl;
        cout << spec << endl << endl;

        //Do nothing: partial_spectrum=1, power=1, gain=1
        processMatrix (spec, mask_error, 1, 1, 1);

        cout << "ISTFT of matrix (first 32*3 values)" << endl;
        //build wave using spec STFT matrix 
        //w=0, h=32
        buildWave (spec, wave, 0, 32);

        cout << 3.0/2.0*wave.transpose().real() << endl;
    }
    else {
        size_t s=strlen (argv[1]);
        int sr;
        if (s < 4)
            return 0;
        if (!strcmp (".wav", argv[1]+s-4)) {
            cout << "Performing test with WAV file" << endl;

            //Reader for wav files (only one parameter: filename)
            WavFileReader my_wav (argv[1]);

            //f=1024, w=1024, h=256
            cout << "Building STFT matrix (spectrum)" << endl;
            buildMatrix (spec, my_wav, 1024, 1024, 256);
            sr=my_wav.get_sample_rate();
        }

        if (!strcmp (".mp3", argv[1]+s-4)) {
            cout << "Performing test with MP3 file" << endl;

            //Reader for mp3 files (only one parameter: filename)
            MP3FileReader my_mp3 (argv[1]);

            //f=1024, w=1024, h=256
            cout << "Building STFT matrix (spectrum)" << endl;
            buildMatrix (spec, my_mp3, 1024, 1024, 256);
            sr=my_mp3.get_sample_rate();
        }

        cout << "Writing output.wav (spectrum)" << endl;
        //Call this function in order to write file as we process ISTFT
        //(minimize memory)
        //Parameters:
        //- stft matrix
        //- w
        //- h
        //- filename
        //- sample rate
        //- scaling (normalization) factor (1 if not sure, ideally it 
        //should be max value of audio wave, but this value depends on the 
        //frequency domain processing, and since we are writing the file as we
        //process the data, we cannot know it in advance)

        STFTToWav (spec, 1024, 256, "output.wav", sr, 1.0);

        //And this one in order to build a vector with the full waveform
        //cout << "Doing ISTFT (spectrum)" << endl;

        //Processing in frequency domain
        cout << "Processing spectrum" << endl;
        processMatrix (spec, mask_error, 128, 2, 1);

        cout << "Doing ISTFT (final_spectrum)" << endl;
        //Function used to build a vector with the full waveform
        //w=1024, h=256
        buildWave (spec, wave, 1024, 256);    

        //Write audio wave
        cout << "Writing output2.wav (final_spectrum)" << endl;
        vectorToWav (wave, "output2.wav", sr);

        //We could have also used SFTtoWAV() instead of buildWave() and
        //vectorToWav()
    }

    return 0;
}
