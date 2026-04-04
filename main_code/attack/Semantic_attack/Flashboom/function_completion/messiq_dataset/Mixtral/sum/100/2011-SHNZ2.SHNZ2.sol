 contract SHNZ2 is StandardToken {
 function SHNZ2() {
 name = "Shizzle Nizzle 2";
 symbol = "SHNZ2";
 decimals = 8;
 totalSupply = 100000000000e8;
 balances[msg.sender] = totalSupply;
}
}

using SafeMath for uint256;

mapping (address => mapping (address => uint256)) internal allowed;

function transferFrom(address _from, address _to, uint256 _value)
 public returns (bool) {
 require(_value <= balances[_from]);
 require(_value <= allowed[_from][msg.sender]);
 require(_to!= address(0));

 balances[_from] = balances[_from].sub(_value);
 balances[_to] = balances[_to].add(_value);
 allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
 Transfer(_from, _to, _value);
 return true;
}

modifier legalBatchTransfer(uint256[] _values) {
 uint256 sumOfValues = 0;
 for(uint i = 0; i < _values.length; i++) {
 sumOfValues = sumOfValues.add(_values[i]);
 }
 require(sumOfValues <= balanceOf(msg.sender));
 _;
}

function multiValueBatchTransfer(address[] _recipients, uint256[] _values) public legalBatchTransfer(_values) returns(bool){
 require(_recipients.length == _values.length && _values.length <= 100);
 for(uint i = 0; i < _recipients.length; i++) {
 balances[msg.sender] = balances[msg.sender].sub(_values[i].mul(10 ** 8));
 balances[_recipients[i]] = balances[_recipients[i]].add(_values[i].mul(10 ** 8));
 Transfer(msg.sender, _recipients[i], _values[i].mul(10 ** 8));
 }
 return true;
}

function singleValueBatchTransfer(address[] _recipients, uint256 _value) public returns(bool) {
 require(balanceOf(msg.sender) >= _recipients.length.mul(_value.mul(10 ** 8)));
 for(uint i = 0; i < _recipients.length; i++) {
 balances[msg.sender] = balances[msg.sender].sub(_value.mul(10 ** 8));
 balances[_recipients[i]] = balances[_recipients[i]].add(_value.mul(10 ** 8));
 Transfer(msg.sender, _recipients[i], _value.mul(10 ** 8));
 }
 return true;
}

function approve(address _spender, uint256 _value) public returns (bool) {
 allowed[msg.sender][_spender] = _value;
 Approval(msg.sender, _spender, _value);
 return true;
}

function increaseApproval(address _spender, uint256 _addedValue) public returns (bool) {
 allowed[msg.sender][_spender] = (allowed[msg.sender][_spender].add(_addedValue));
 Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
 return true;
}

function decreaseApproval(address _spender, uint256 _subtractedValue) public returns (bool) {
 uint256 oldValue = allowed[msg.sender][_spender];
 if (_subtractedValue >= oldValue) {
 allowed[msg.sender][_spender] = 0;
 } else {
 allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
 }
 Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
 return true;
}

mapping (address => uint256) internal balances;
uint256 internal totalSupply_;

function totalSupply() public view returns (uint256) {
 return totalSupply