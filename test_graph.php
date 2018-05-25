<?php
require_once("TensorFlow.php");
require_once("PrintGraph.php");

$tf = new TensorFlow();
$ret =
	$tf->add(
		$tf->add(
			$tf->placeholder("x", \TF\DOUBLE),
			$tf->placeholder("y", \TF\DOUBLE)),
		$tf->constant(0.5));
print_graph($tf->graph);

$def = $tf->graph->export();

$tf = new TensorFlow();
print_graph($tf->graph);

$tf->graph->import($def, "import");
print_graph($tf->graph);


$x = $tf->tensor([ord('a'),ord('b'),ord('c'),ord('d')], \TF\INT8);
var_dump($x->value());
var_dump($x->bytes());
$x->setBytes("xyz_");
var_dump($x->value());
