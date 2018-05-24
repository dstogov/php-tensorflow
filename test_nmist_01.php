<?php
/* 
 * Handwritten digits recognation using pre-traind saved model
 *
 * See: http://yann.lecun.com/exdb/mnist/
 *
 */
require_once("TensorFlow.php");

const MODEL       = './models/mnist_500_500_softmax_linear';
const TEST_IMAGES = './datasets/mnist/t10k-images-idx3-ubyte.gz';
const TEST_LABELS = './datasets/mnist/t10k-labels-idx1-ubyte.gz';

function main() {
	$tf = new TensorFlow();
	/* Load Session */
	$sess = $tf->loadSavedModel(MODEL);
	/* Output of the loaded Neural Networ */
	$out = $tf->graph->operation('softmax_linear/add')->output(0);
	/* Index of the top value */
	$out_label =
		$tf->op('Reshape', [
			$tf->op('TopKV2', [$out, $tf->constant(1, \TF\INT32)], [], [], null, 1),
			$tf->constant([-1])]);
	/* Recognation */
	$ret = $sess->run($out_label, ['Placeholder' => get_images($tf)]);
	/* Verification */
	$labels = $ret->value();
	$right = verify_labels($labels);
	printf("Num examples: %d Num correct: %d  Precision: %g\n",
		count($labels), $right, $right / count($labels));
}

function get_images($tf) {
	$f = gzopen(TEST_IMAGES, 'r');
	$magic = read32($f);
	if ($magic != 2051) {
		throw new Exception("Invalid magic number '$magic' in MNIST image file");
	}
	$num_images = read32($f);
	$rows = read32($f);
	$cols = read32($f);
	$s = gzread($f, $rows * $cols * $num_images);
	gzclose($f);
	$pixels = $rows * $cols;
	$n = 0;
	$ret = $tf->tensor(null, \TF\FLOAT, [$num_images, $pixels]);
	$data = $ret->data();
	for ($i = 0; $i < $num_images; $i++) {
		for ($j = 0; $j < $pixels; $j++) {
			$data[$i * $pixels + $j] = ord($s[$n++]) / 255.0;
		}
	}
	return $ret;		
}

function read32($f) {
	return unpack("N", gzread($f, 4))[1];
}

function get_labels($tf) {
	$f = gzopen(TEST_LABELS, 'r');
	$magic = read32($f);
	if ($magic != 2049) {
		throw new Exception("Invalid magic number '$magic' in MNIST label file");
	}
	$num_items = read32($f);
	$s = gzread($f, $num_items);
	gzclose($f);
	$ret = $tf->tensor(null, \TF\INT32, [$num_items]);
	$data = $ret->data();
	for ($i = 0; $i < $num_items; $i++) {
		$data[$i] = ord($s[$i]);
	}
	return $ret;
}

function verify_labels($labels) {
	$right = 0;
	$f = gzopen(TEST_LABELS, 'r');
	$magic = read32($f);
	if ($magic != 2049) {
		throw new Exception("Invalid magic number '$magic' in MNIST label file");
	}
	$num_items = read32($f);
	$s = gzread($f, $num_items);
	gzclose($f);
	for ($i = 0; $i < $num_items; $i++) {
		if (ord($s[$i]) == $labels[$i]) {
			$right++;
		}
	}
	return $right;
}

main();
