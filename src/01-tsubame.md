
# TSUBAME3.0でChainerMN

@<icon>{yousei} 「そもそもTSUBAMEっていうのは、東京工業大学に設置された大規模クラスター型スーパーコンピュータのことなんだ。」

@<icon>{cheita} 「1ノードあたりNVIDIA P100を4枚、計540ノードで国内有数のスパコン！単精度で24.3PFlopsで~~」

@<icon>{cheiko}「(ちぇい太君の知識自慢が始まってしまった・・・)」

## TSUBAME3.0でChainerMN
### TSUBAME3.0へログイン
TODO

### Chainerのインストール
まず、Chainerのインストールを行います。インタラクティブジョブを起動し、以下のコマンドを打ち込みます

```
xxxxx@login0:~> GROUP="自分のグループ名"
xxxxx@login0:~> qrsh -g $GROUP -l s_gpu=1 -l h_rt=1:00:00
xxxxx@r8i6n8:~> . /etc/profile.d/modules.sh
xxxxx@r8i6n8:~> module load python/3.6.5
xxxxx@r8i6n8:~> module load cuda/9.2.148
xxxxx@r8i6n8:~> module load openmpi/2.1.2-opa10.9
xxxxx@r8i6n8:~> pip install --user mpi4py cupy-cuda92==6.2.0 chainer==6.2.0 
xxxxx@r8i6n8:~> python -c 'import chainer; chainer.print_runtime_info()'
Platform: Linux-4.4.121-92.85-default-x86_64-with-SuSE-12-x86_64
Chainer: 6.2.0
NumPy: 1.17.0
CuPy:
  CuPy Version          : 6.2.0
  CUDA Root             : /apps/t3/sles12sp2/cuda/9.2.148
  CUDA Build Version    : 9020
  CUDA Driver Version   : 10000
  CUDA Runtime Version  : 9020
  cuDNN Build Version   : 7402
  cuDNN Version         : 7402
  NCCL Build Version    : 2402
  NCCL Runtime Version  : 2402
iDeep: Not Available
```

注意事項
 * 環境は、特段の理由がない限り、CUDA 9.2, Open MPI 2.1.2 が推奨されます。
 * Chainerのバージョンは、6.2.0 を推奨します。また、明示的にインストールするバージョンを指定したほうが安全です
 * CuPYは、`cupy-cuda92` というバイナリインストールを用いるのが推奨です。単に`cupy`と指定するとソースからのビルドになりますが、その場合、NCCLとCuDNNを自分でインストールしてした上でビルドする必要が出てきます。

### スクリプト(MNIST)の準備

ChainerMNプログラムを分散環境で動かすには、動かしたいプログラムとデータに加えて、2つのファイルを用意します。

 * 実行スクリプト `train.py`
 * 動かしたいアプリケーションを起動するジョブスクリプト（ここでは例として以下の`job_mnist.sh`)
 * 補助ファイル `run.sh` 
 
ジョブスクリプトの内容は、走らせたいジョブの処理の内容に従って細かく変更を加えていきます。
補助ファイルは、最初に作成したあとは基本的に変更の必要はありません


ChainerのMNISTサンプルは、初回実行時に`HOME`ディレクトリにMNISTデータをダウンロードする。計算ノードからはインターネットにアクセスできないので、ログインノードでMNISTを一回実行してデータをダウンロードさせる。この実行はChainerMNである必要はない。（まちがえてmasterブランチのtrain_mnist.pyをダウンロードするとエラーで実行できないので注意）
（なお、手動でデータをコピーしても良い）

```
# NOTE: env.shの実行を忘れないように
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/examples/mnist/train_mnist.py -O train_mnist_single.py
$ python train_mnist_single.py -e 1
```

次に、ChainerMN用の train_mnist.py をダウンロードする。（まちがえてmasterブランチのtrain_mnist.pyをダウンロードするとエラーで実行できないので注意）

```
$ wget https://raw.githubusercontent.com/chainer/chainer/v5.1.0/examples/chainermn/mnist/train_mnist.py
```


### ジョブスクリプトの準備
次に、テストとしてMNISTを並列学習するジョブを投入してみます。以下の `job_mnist.sh` と `run.sh` の2つのファイルを用意して、実行権限を付与します

```
xxxxx@login0:~> vi job_mnist.sh
xxxxx@login0:~> chmod +x job_mnist.sh

xxxxx@login0:~> vi run.sh
xxxxx@login0:~> chmod +x run.sh
```

ジョブを投入します。

```
xxxxx@login0:~> qrsh -g $GROUP -l s_gpu=4 -l h_rt=1:00:00 ./job_mnist.sh
Sat Aug  3 21:37:25 JST 2019
Platform: Linux-4.4.121-92.85-default-x86_64-with-SuSE-12-x86_64
Chainer: 6.2.0
NumPy: 1.17.0
CuPy:
  CuPy Version          : 6.2.0
  CUDA Root             : /apps/t3/sles12sp2/cuda/9.2.148
  CUDA Build Version    : 9020
  CUDA Driver Version   : 10000
  CUDA Runtime Version  : 9020
  cuDNN Build Version   : 7402
  cuDNN Version         : 7402
  NCCL Build Version    : 2402
  NCCL Runtime Version  : 2402
iDeep: Not Available
r1i7n6 2 all.q@r1i7n6 <NULL>
r2i3n1 2 all.q@r2i3n1 <NULL>
r8i6n8 2 all.q@r8i6n8 <NULL>
r2i4n8 2 all.q@r2i4n8 <NULL>
==========================================
Num process (COMM_WORLD): 4
Using GPUs
Using pure_nccl communicator
Num unit: 1000
Num Minibatch-size: 100
Num epoch: 2
==========================================
--------------------------------------------------------------------------
A process has executed an operation involving a call to the
"fork()" system call to create a child process.  Open MPI is currently
operating in a condition that could result in memory corruption or
other system errors; your job may hang, crash, or produce silent
data corruption.  The use of fork() (or system() or other calls that
create child processes) is strongly discouraged.

The process that invoked fork was:

  Local host:          [[19872,1],0] (PID 21643)

If you are *absolutely sure* that your application will successfully
and correctly survive a call to fork(), you may disable this warning
by setting the mpi_warn_on_fork MCA parameter to 0.
--------------------------------------------------------------------------
epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
1           0.289679    0.118232              0.915267       0.9628                    4.36749
2           0.0913869   0.0768252             0.972          0.9755                    5.81177
```

出力された結果を見て、正しく実行されたかどうか確認しましょう。
 * 冒頭に `chainer.print_runtime_info()` の結果が出力されているので、ライブラリが正しく読み込まれていることを確認します
 * `-l s_gpu=4` という指定をしましたので、合計4プロセスで実行されます。 出力の`Num process (COMM_WORLD): 4` と一致していることを確認します。
 * コミュニケーターとして `pure_nccl communicator` が使われていることを確認します
 * MNISTが正しく学習できていることを確認します （注： なお、MNISTã¯計算負荷が軽いので、複数GPU実行ではむしろ遅くなるケースが多いです）

以上でChainerMNプログラムを分散学習できました


## TSUBAME3.0の利用申請

## おわりに
- TODO: TSUBAME簡単につかえるよ。ハッカソンもしているよ
