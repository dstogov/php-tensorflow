# Привязка TensorFlow к PHP

Это экспериментальная (а ещё устаревшая) привязка к библиотеке [TensorFlow](https://www.tensorflow.org), написанная на PHP с использованием расширения [FFI](https://github.com/dstogov/php-ffi).

## Установка

Расширение работает на максимальной версии библиотеки TensorFlow 2.3.4

```bash
FILENAME=libtensorflow-cpu-linux-x86_64-2.3.4.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
sudo tar -C /usr/local -xzf ${FILENAME}
# Опционально
sudo ldconfig /usr/local/lib
```
