<?php
require_once("TensorFlow.php");

function print_graph($g) {
	foreach ($g->operations() as $op) {
		echo $op->name() . ": " . $op->type() . "\n";
		$count = $op->numInputs();
		for ($i = 0; $i < $count; $i++) {
			$in = $op->input($i);
			$out = $in->producer();
			echo "  in_$i: " . $in->type($i) .
				", from " . $out->op()->name() . "/out_" . $out->index() . "\n";
		}
		$count = $op->numOutputs();
		for ($i = 0; $i < $count; $i++) {
			$out = $op->output($i);
			$num = $out->numConsumers();
			$s = "";
			if ($num) {
				$inputs = $out->consumers();
				$s = ", to (";
				$first = true;
				foreach ($inputs as $in) {
					if (!$first) {
						$s = ", ";
					} else {
						$first = false;
					}
					$s .= $in->op()->name() ."/" . $in->index();
				}
				$s .= ")";
			}
			echo "  out_$i: " . $out->type($i) . "$s\n";
		}
	}
	echo "\n";
}

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
