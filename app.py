from baseline import *
import pickle
from flask import Flask, render_template
import uuid

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", cname='none')

@app.route('/mozart')
def mozart():
    with open('Mozart.pkl', 'rb') as f:
        generator = pickle.load(f)
    filename = generate(generator)
    return render_template("index.html", cname='mozart', filename=filename)

@app.route('/beethoven')
def beethoven():
    with open('Beethoven.pkl', 'rb') as f:
        generator = pickle.load(f)
    filename = generate(generator)
    return render_template("index.html", cname='beethoven', filename=filename)

@app.route('/others')
def others():
    with open('Classical_Era_further_epoch_10000.pkl', 'rb') as f:
        generator = pickle.load(f)
    filename = generate(generator)
    return render_template("index.html", cname='others', filename=filename)

def generate(generator):
    generator.eval()
    sample_latent = torch.randn(n_samples, latent_dim)
    if torch.cuda.is_available():
        generator = generator.cuda()
        sample_latent = sample_latent.cuda()
    samples = generator(sample_latent).cpu().detach().numpy()
    samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(zip(programs, is_drums, track_names)):
        pianoroll = np.pad(samples[idx] > 0.5,((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches)))
        tracks.append(BinaryTrack(name=track_name,program=program,is_drum=is_drum,pianoroll=pianoroll))
    m = Multitrack(tracks=tracks,tempo=tempo_array,resolution=beat_resolution)
    filename = f'{uuid.uuid4()}.mid'
    m.write(f'static/{filename}')
    return filename
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)