 contract TokenFactory is StandardToken {
 string public name;
 string public symbol;
 uint256 public decimals;

 function TokenFactory(uint256 _initialAmount, string memory _tokenName, uint8 _decimalUnits, string memory _tokenSymbol) {
 balances[msg.sender] = _initialAmount;
 totalSupply = _initialAmount;
 name = _tokenName;
 decimals = _decimalUnits;
 symbol = _tokenSymbol;
 }
}

using SafeMath for uint256;

mapping (address => mapping (address => uint256)) allowed;

function transferFrom(address _from, address _to, uint256 _value) returns (bool) {
 var _allowance = allowed[_from][msg.sender];

 balances[_to] = balances[_to].add(_value);
 balances[_from] = balances[_from].sub(_value);
 allowed[_from][msg.sender] = _allowance.sub(_value);
 Transfer(_from, _to, _value);
 return true;
}

function approve(address _spender, uint256 _value) returns (bool) {
 require((_value == 0) || (allowed[msg.sender][_spender] == 0));

 allowed[msg.sender][_spender] = _value;
 Approval(msg.sender, _spender, _value);
 return true;
}

function allowance(address _owner, address _spender) constant returns (uint256 remaining) {
 return allowed[_owner][_spender];
}

mapping(address => uint256) balances;

function transfer(address _to, uint256 _value) returns (bool) {
 balances[msg.sender] = balances[msg.sender].sub(_value);
 balances[_to] = balances[_to].add(_value);
 Transfer(msg.sender, _to, _value);
 return true;
}

function balanceOf(address _owner) constant returns (uint256 balance) {
 return balances[_owner];
}

event Transfer(address indexed from, address indexed to, uint256 value);
event Approval(address indexed owner, address indexed spender, uint256 value);

library SafeMath {
 function mul(uint256 a, uint256 b) internal constant returns (uint256) {
 uint256 c = a * b;
 assert(a == 0 || c / a == b);
 return c;
 }

 function div(uint256 a, uint256 b) internal constant returns (uint256) {
 uint256 c = a / b;
 return c;
 }

 function sub(uint256 a, uint256 b) internal constant returns (uint256) {
 assert(b <= a);
 return a - b;
 }

 function add(uint256 a, uint256 b) internal constant returns (uint256) {
 uint256 c = a + b;
 assert(c >= a);
 return c;
 }
}