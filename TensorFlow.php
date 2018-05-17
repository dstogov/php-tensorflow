<?php
namespace TF;

use FFI;

if (!extension_loaded("FFI")) {
	die("FFI extension required\n");
}

const FLOAT      = 1;
const DOUBLE     = 2;
const INT32      = 3;
const UINT8      = 4;
const INT16      = 5;
const INT8       = 6;
const STRING     = 7;
const COMPLEX64  = 8;
const COMPLEX    = 8;
const INT64      = 9;
const BOOL       = 10;
const QINT8      = 11;
const QUINT8     = 12;
const QINT32     = 13;
const BFLOAT16   = 14;
const QINT16     = 15;
const QUINT16    = 16;
const UINT16     = 17;
const COMPLEX128 = 18;
const HALF       = 19;
const RESOURCE   = 20;
const VARIANT    = 21;
const UINT32     = 22;
const UINT64     = 23;

const OK                  = 0;
const CANCELLED           = 1;
const UNKNOWN             = 2;
const INVALID_ARGUMENT    = 3;
const DEADLINE_EXCEEDED   = 4;
const NOT_FOUND           = 5;
const ALREADY_EXISTS      = 6;
const PERMISSION_DENIED   = 7;
const UNAUTHENTICATED     = 16;
const RESOURCE_EXHAUSTED  = 8;
const FAILED_PRECONDITION = 9;
const ABORTED             = 10;
const OUT_OF_RANGE        = 11;
const UNIMPLEMENTED       = 12;
const INTERNAL            = 13;
const UNAVAILABLE         = 14;
const DATA_LOSS           = 15;

class API {
	static protected $ffi;
	static protected $tensor_ptr;

	static protected function init_tf_ffi() {
		self::$ffi = FFI::load(__DIR__ . "/tf_api.h");
		self::$tensor_ptr = self::$ffi->type("TF_Tensor*");
	}

	public function version() {
		return (string)self::$ffi->TF_Version();
	}
}

final class Status extends API {
	public $c;

	public function __construct() {
		$this->c = self::$ffi->TF_NewStatus();
	}

	public function __destruct() {
		self::$ffi->TF_DeleteStatus($this->c);
	}

	public function code() {
		return (int)self::$ffi->TF_GetCode($this->c);
	}

	public function string() {
		return (string)self::$ffi->TF_Message($this->c);
	}

	public function error() {
		return $this->string();
	}
}

final class Tensor extends API {
	public $c;
	private $dataType;
	private $ndims;
	private $shape;
	private $nflattened;
	private $dataSize;
	private $status;

	public function init($value, $dataType = null, $shape = null, $status = null) {
    	if (is_null($status)) {
	    	$status = new Status();
		}
		$this->status = $status;

		if ($dataType == null) {
			$dataType = self::_guessType($value);
		}
		if ($shape == null) {
			$shape = self::_guessShape($value);
		}
		$ndims = 0;
		$shapePtr = null;
		$nflattened = 1;
		if (is_array($shape)) {
			$ndims = count($shape);
			if ($ndims > 0) {
				$shapePtr = self::$ffi->new("int64_t[$ndims]");
				$i = 0;
				foreach ($shape as $val) {
					$shapePtr[$i] = $val;
					$nflattened *= $val;
					$i++;
				}
			}
		}
		if ($dataType == STRING) {
			$nbytes = $nflattened * 8 + self::_byteSizeOfEncodedStrings($value);
		} else {
			$nbytes = self::$ffi->TF_DataTypeSize($dataType) * $nflattened;
		}

		$this->c = self::$ffi->TF_AllocateTensor($dataType, $shapePtr, $ndims, $nbytes);
		$this->dataType = $dataType;
		$this->shape = $shape;
		$this->ndims = $ndims;
		$this->nflattened = $nflattened;
		$this->dataSize = $nbytes;

		$data = $this->data();
		if ($dataType == STRING) {
			$this->_stringEncode($value, $data);
		} else {
			$this->_encode($value, $data);
		}
	}

	public function initFromC($cdata) {
		if (is_null($this->status)) {
			$this->status = new Status();
		}

		$this->c = $cdata;
		$this->dataType = self::$ffi->TF_TensorType($cdata);
		$ndims = self::$ffi->TF_NumDims($cdata);
		$this->ndims = $ndims;
		$this->nflattened = 1;
		for ($i = 0; $i < $ndims; $i++) {
			$dim = self::$ffi->TF_Dim($cdata, $i);
			$this->shape[$i] = $dim;
			$this->nflattened *= $dim;
		}
		$this->dataSize = self::$ffi->TF_TensorByteSize($cdata);
	}

	public function __destruct() {
		if (!is_null($this->c)) {
			self::$ffi->TF_DeleteTensor($this->c);
		}
	}

	public function dataType() {
		return $this->dataType;
	}

	public function shape() {
		return $this->shape;
	}

	public function value() {
		$data = $this->data();
		if ($this->dataType == STRING) {
			return $this->_stringDecode($data);
		} else {
			return $this->_decode($data);
		}
	}

	public function isSerializable() {
		throw new \Exception("Not Implemented"); //???
	}

	public function write() {
		throw new \Exception("Not Implemented"); //???
	}

	public static function read() {
		throw new \Exception("Not Implemented"); //???
	}

	public function plainData() {
		return self::$ffi->TF_TensorData($this->c);
	}

	public function data() {
		static $map = [
			FLOAT      => "float",
			DOUBLE     =>"double",
			INT32      => "int32_t",
			UINT8      => "uint8_t",
			INT16      => "int16_t",
			INT8       => "int8_t",
			COMPLEX64  => null,
			COMPLEX    => null,
			INT64      => "int64_t",
			BOOL       => "bool",
			QINT8      => null,
			QUINT8     => null,
			QINT32     => null,
			BFLOAT16   => null,
			QINT16     => null,
			QUINT16    => null,
			UINT16     => "uint16_t",
			COMPLEX128 => null,
			HALF       => null,
			RESOURCE   => null,
			VARIANT    => null,
			UINT32     => "uint32_t",
			UINT64     => "uint64_t",
		];
		$n = $this->nflattened;
		if ($this->dataType == STRING) {
			$m = $this->dataSize - $this->nflattened * 8;
			return self::$ffi->cast(
				"struct {uint64_t offsets[$n]; char data[$m];}",
				$this->plainData(), true);
		} else {
			$cast = @$map[$this->dataType()];
			if (isset($cast)) {
				$cast .= "[$n]";
				return self::$ffi->cast($cast, $this->plainData(), true);
			} else {
				throw new \Exception("Not Implemented"); //???
			}
		}
	}

	private function _stringEncode($value, $data, &$offset = 0, &$dim_offset = 0, $dim = 0, $n = 0) {
		if ($dim < $this->ndims) {
			$n = $this->shape[$dim];
			if (!is_array($value) || count($value) != $n) {
				throw new \Exception("value/shape mismatch");
			}
			$dim++;
			$i = 0;
			foreach ($value as $val) {
				$this->_stringEncode($val, $data, $offset, $dim_offset, $dim, $i);
				$i++;
			}
			return;
		}

		$str = (string)$value;
		$data->offsets[$dim_offset++] = $offset;
		$offset += self::$ffi->TF_StringEncode(
			$str,
			strlen($str),
			FFI::offset($data->data, $offset),
			self::$ffi->TF_StringEncodedSize(strlen($str)),
			$this->status->c);
		if ($this->status->code() != OK) {
			throw new \Exception($this->status->error());
		}
	}

	private function _stringDecode($data, &$dim_offset = 0, $dim = 0, $n = 0) {
		if ($dim < $this->ndims) {
			$n = $this->shape[$dim];
			$dim++;
			$ret = array();
			for ($i = 0; $i < $n; $i++) {
				$ret[$i] = $this->_stringDecode($data, $dim_offset, $dim, $i);
			}
			return $ret;
		}

		$offset = $data->offsets[$dim_offset++];

		$dst = self::$ffi->new("char*");
		$dst_len = self::$ffi->new("size_t");
		self::$ffi->TF_StringDecode(
				FFI::offset($data->data, $offset),
				$this->dataSize - $offset,
				FFI::addr($dst),
				FFI::addr($dst_len),
				$this->status->c);
		if ($this->status->code() != OK) {
			throw new \Exception($this->status->error());
		}
		return FFI::string($dst, (int)$dst_len);
	}

	private function _encode($value, $data, &$dim_offset = 0, $dim = 0, $n = 0) {
		if ($dim < $this->ndims) {
			$n = $this->shape[$dim];
			if (!is_array($value) || count($value) != $n) {
				throw new \Exception("value/shape mismatch");
			}
			$dim++;
			$i = 0;
			foreach ($value as $val) {
				$this->_encode($val, $data, $dim_offset, $dim, $i++);
			}
			return;
		}
		$data[$dim_offset++] = $value;
	}

	private function _decode($data, &$dim_offset = 0, $dim = 0, $n = 0) {
		if ($dim < $this->ndims) {
			$n = $this->shape[$dim];
			$dim++;
			$ret = array();
			for ($i = 0; $i < $n; $i++) {
				$ret[$i] = $this->_decode($data, $dim_offset, $dim, $i);
			}
			return $ret;
		}
		return $data[$dim_offset++];
	}

	private static function _guessType($value) {
		if (is_array($value)) {
			foreach($value as $val) {
				return self::_guessType($val);
			}
		}
		if (is_int($value)) {
			return PHP_INT_SIZE == 4 ? INT32 : INT64;
		} else if (is_double($value)) {
			return DOUBLE;
		} else if (is_bool($value)) {
			return BOOL;
		} else if (is_string($value)) {
			return STRING;
	    } else {
    		throw new \Exception("Cannot guess type");
        }
	}

	private static function _guessShape($value, array $shape = []) {
		if (is_array($value)) {
			$shape[] = count($value);
			foreach($value as $val) {
				return self::_guessShape($val, $shape);
			}
		}
		return $shape;
	}

	private static function _byteSizeOfEncodedStrings($value) {
		if (is_array($value)) {
			$size = 0;
			foreach($value as $val) {
				$size += self::_byteSizeOfEncodedStrings($val);
			}
			return $size;
		} else {
			$val = (string)$value;
			return self::$ffi->TF_StringEncodedSize(strlen($val));
		}
	}
}

final class Input extends API {
	public $c;

	public function init(Operation $operation, int $index) {
		$this->c = self::$ffi->new("TF_Input");
		$this->c->oper = $operation->c;
		$this->c->index = $index;
	}

	public function initFromC($cdata) {
		$this->c = $cdata;
	}

	public function op() {
		$op = new Operation();
		$op->initFromC($this->c->oper);
		return $op;
	}

	public function index() {
		return $this->c->index;
	}

	public function type() {
		return (int)self::$ffi->TF_OperationInputType($this->c);
	}

	public function producer() {
		$cdata = self::$ffi->TF_OperationInput($this->c);
		$output = new Output();
		$output->initFromC($cdata);
		return $output;
	}
}

final class Output extends API {
	public $c;

	public function init(Operation $operation, int $index) {
		$this->c = self::$ffi->new("TF_Output");
		$this->c->oper = $operation->c;
		$this->c->index = $index;
	}

	public function initFromC($cdata) {
		$this->c = $cdata;
	}

	public function op() {
		$op = new Operation();
		$op->initFromC($this->c->oper);
		return $op;
	}

	public function index() {
		return $this->c->index;
	}

	public function type() {
		return (int)self::$ffi->TF_OperationOutputType($this->c);
	}

	public function shape() {
		throw new \Exception("Not Implemented"); //???
	}
}

final class Operation extends API {
	public $c;

	public function init($graph, $type, $name, array $input = [], array $control = [], array $attr = []) {
		$status = new Status();
		$desc = self::$ffi->TF_NewOperation($graph->c, $type, $name);

		foreach ($input as $in) {
			if ($in instanceof Output) {
				self::$ffi->TF_AddInput($desc, $in->c);
			} else if ($input instanceof OutputList) {
				throw new \Exception("Not Implemented"); //???
				//self::$ffi->TF_AddInputList($desc, ...);
			} else {
				throw new \Exception("Not Implemented"); //???
			}
		}

		foreach ($control as $ctl) {
			self::$ffi->TF_AddControlInput($desc, $ctl->c);
		}

		// TODO: proper $attr support ???
		foreach ($attr as $key => $val) {
			switch ($key) {
				case "value":
					self::$ffi->TF_SetAttrTensor($desc, $key, $val->c, $status->c);
					if ($status->code() != OK) {
						throw new \Exception($status->error());
					}
					break;
				case "dtype":
					self::$ffi->TF_SetAttrType($desc, $key, $val);
					break;
				default:
					throw new \Exception("Unknown Operation attr");
			}
		}

		$this->c = self::$ffi->TF_FinishOperation($desc, $status->c);
		if ($status->code() != OK) {
			throw new \Exception($status->error());
		}
	}

	public function initFromC($cdata) {
		$this->c = $cdata;
	}

	public function name() {
		return (string)self::$ffi->TF_OperationName($this->c);
	}

	public function type() {
		return (string)self::$ffi->TF_OperationOpType($this->c);
	}

	public function numInputs() {
		return (int)self::$ffi->TF_OperationNumInputs($this->c);
	}

	public function inputType($n) {
		$input = self::$ffi->new("TF_Input");
		$input->oper = $this->c;
		$input->index = $n;
		return (int)self::$ffi->TF_OperationInputType($input);
	}

	public function numOutputs() {
		return (int)self::$ffi->TF_OperationNumOutputs($this->c);
	}

	public function outputType($n) {
		$output = self::$ffi->new("TF_Output");
		$output->oper = $this->c;
		$output->index = $n;
		return (int)self::$ffi->TF_OperationOutputType($output);
	}

	public function outputListSize($name) {
		$status = new Status();
		$ret = (int)self::$ffi->TF_OperationOutputListLength($this->c, $name, $status->c);
		if ($status->code() != OK) {
			throw new \Exception($status->error());
		}
		return ret;
	}

	public function input($n) {
		$input = new Input();
		$input->init($this, $n);
		return $input;
	}

	public function output($n) {
		$output = new Output();
		$output->init($this, $n);
		return $output;
	}
}

final class Graph extends API {
	public $c;

	public function __construct() {
		$this->c = self::$ffi->TF_NewGraph();
	}

	public function __destruct() {
		self::$ffi->TF_DeleteGraph($this->c);
	}

	public function operation(string $name) {
		$cdata = self::$ffi->TF_GraphOperationByName($this->c, $name);
		if (is_null($cdata)) {
			return null;
		}
		$op = new Operation();
		$op->initFromC($cdata);
		return $op;
	}

	public function operations() {
		$pos = self::$ffi->new("size_t[1]");
		$pos[0] = 0;
		$ops = [];
		while (1) {
			$cdata = self::$ffi->TF_GraphNextOperation($this->c, $pos);
			if (is_null($cdata)) {
				break;
			}
			$op = new Operation();
			$op->initFromC($cdata);
			$ops[] = $op;
		}
		return $ops;
	}

	public function addOperation($type, $name, array $input = [], array $control = [], array $attr = []) {
		$op = new Operation();
		$op->init($this, $type, $name, $input, $control, $attr);
		return $op;
	}

	public function write() {
		throw new \Exception("Not Implemented"); //???
	}

	public static function import() {
		throw new \Exception("Not Implemented"); //???
	}
}

final class SessionOptions extends API {
	public $c;

	public function __construct() {
		$this->c = self::$ffi->TF_NewSessionOptions();
	}
}

final class Session extends API {
	private $c;
	private $graph;
	private $options;
	private $status;

	public function __construct(Graph $graph, SessionOptions $options = null, Status $status = null) {
		$this->graph = $graph;
	    if (is_null($options)) {
    		$options = new SessionOptions();
		}
		$this->options = $options;
	    if (is_null($status)) {
    		$status = new Status();
		}
		$this->status = $status;
		$this->c = self::$ffi->TF_NewSession($this->graph->c, $this->options->c, $this->status->c);
		if ($this->status->code() != OK) {
			throw new \Exception($this->status->error());
		}
	}

	public function __destruct() {
		$this->close();
	}

	public function close() {
		if (!is_null($this->c)) {
			self::$ffi->TF_CloseSession($this->c, $this->status->c);
			if ($this->status->code() != OK) {
				throw new \Exception($this->status->error());
			}
			self::$ffi->TF_DeleteSession($this->c, $this->status->c);
			$this->c = null;
		}
	}

	public function run($fetches = null, array $feeds = null, $targes = null) {
		$n_fetches = 0;
		$c_fetches = null;
		$c_fetchTensors = null;
		if (!is_null($fetches)) {
			if (is_array($fetches)) {
				$n_fetches = count($fetches);
				if ($n_fetches > 0) {
					$c_fetches = self::$ffi->new("TF_Output[$n_fetches]");
					$t_fetchTensors = self::$ffi->type(self::$tensor_ptr, [$n_fetches]);
					$c_fetchTensors = self::$ffi->new($t_fetchTensors);
				}
				$i = 0;
				foreach ($fetches as $fetch) {
					$c_fetches[$i] = $fetch->c;
					$i++;
				}
			} else {
				$n_fetches = 1;
				$c_fetches = self::$ffi->new("TF_Output[1]");
				$t_fetchTensors = self::$ffi->type(self::$tensor_ptr, [$n_fetches]);
				$c_fetchTensors = self::$ffi->new($t_fetchTensors);
				$c_fetches[0] = $fetches->c;
			}
		}

		$n_feeds = 0;
		$c_feeds = null;
		$c_feedTensors = null;
		if (is_array($feeds)) {
			$n_feeds = count($feeds);
			if ($n_feeds > 0) {
				$c_feeds = self::$ffi->new("TF_Output[$n_feeds]");
				$c_feedTensors = self::$ffi->new("TF_Tensor*[$n_feeds]");
				$i = 0;
				foreach ($feeds as $key => $val) {
					$op = $this->graph->operation($key);
					if (!is_null($op)) {
						$feed = new Output();
						$feed->init($op, 0);
						$c_feeds[$i] = $feed->c;
						$c_feedTensors[$i] = $val->c;
						$i++;
					} else {
						--$n_feeds;
					}
				}
			}
		}

		$n_targets = 0;
		$c_targets = null;
		// TODO: process $targets ???

		self::$ffi->TF_SessionRun($this->c, null,
			$c_feeds, $c_feedTensors, $n_feeds, // Inputs
			$c_fetches, $c_fetchTensors, $n_fetches, // Outputs
			$c_targets, $n_targets, // Operations
			null, $this->status->c);

		if ($this->status->code() != OK) {
			throw new \Exception($this->status->error());
		}

		if (is_array($fetches)) {
			$ret = array();
			for ($i = 0; $i < $n_fetches; $i++) {
				$t = new Tensor();
				$t->initFromC($c_fetchTensors[$i]);
				$ret[$i] = $t;
			}
			return $ret;
		} else if (!is_null($fetches)) {
			$t = new Tensor();
			$t->initFromC($c_fetchTensors[0]);
			return $t;
		}
	}
}

final class TensorFlow extends API {
	private $graph;
	private $status;
	static private $num = 0;

	public function __construct() {
		if (is_null(self::$ffi)) self::init_tf_ffi();
	}

	public function loadSavedModel() {
		throw new \Exception("Not Implemented"); //???
	}

	public function tensor($value, $dataType = null, $shape = null) {
		$status = $this->_defaultStatus();
		$tensor = new Tensor();
		$tensor->init($value, $dataType, $shape, $status);
		return $tensor;
	}

	public function constant($value, $dataType = null, $shape = null, $name = null) {
		if (is_null($name)) {
			$name = self::_genName("Const");
		}
		$graph = $this->_defaultGraph();
		$status = $this->_defaultStatus();
		$tensor = new Tensor();
		$tensor->init($value, $dataType, $shape, $status);
		$op = $graph->addOperation("Const", $name, [], [],
			[	"dtype" => $tensor->dataType(),
				"value" => $tensor,
			]);
		return $op->output(0);
	}

	public function placeholder($name, $dataType) {
		if (is_null($name)) {
			$name = self::_genName("Const");
		}
		$graph = $this->_defaultGraph();
		$op = $graph->addOperation("Placeholder", $name, [], [],
			["dtype" => $dataType]);
		return $op->output(0);
	}

	public function add($x, $y, $name = null) {
		if (is_null($name)) {
			$name = self::_genName("Add");
		}
		$graph = $this->_defaultGraph();
		$op = $graph->addOperation("Add", $name, [$x, $y]);
		return $op->output(0);
	}

	public function session() {
		$graph = $this->_defaultGraph();
		$status = $this->_defaultStatus();
		return new Session($graph, null, $this->status);
	}

	public function operations() {
		$graph = $this->_defaultGraph();
		return $graph->operations();
	}

	protected function _defaultGraph() {
		if (!isset($this->graph)) {
			$this->graph = new Graph();
		}
		return $this->graph;
	}

	protected function _defaultStatus() {
		if (!isset($this->status)) {
			$this->status = new Status();
		}
		return $this->status;
	}

	static protected function _genName($name) {
		$name .= "_" . ++self::$num;
		return $name;
	}
}

class_alias("\TF\TensorFlow", "TensorFlow");
