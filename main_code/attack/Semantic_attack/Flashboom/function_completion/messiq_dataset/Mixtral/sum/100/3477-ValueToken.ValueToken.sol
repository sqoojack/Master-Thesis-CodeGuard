 contract ValueToken is ValueTokenBase {

 function _transfer(address _from, address _to, uint _value) internal returns (bool) {
 // Prevent transfer to 0x0 address. Use burn() instead
 require(_to!= 0x0);
 // Check if the sender has enough
 require(balances[_from] >= _value);
 // Check for overflows
 require(balances[_to] + _value > balances[_to]);
 // Save this for an assertion in the future
 uint previousBalances = balances[_from] + balances[_to];
 // Subtract from the sender
 balances[_from] -= _value;
 // Add the same to the recipient
 balances[_to] += _value;
 Transfer(_from, _to, _value);
 // Asserts are used to use static analysis to find bugs in your code. They should never fail
 assert(balances[_from] + balances[_to] == previousBalances);

 return true;
 }

 function transfer(address _to, uint256 _value) public returns (bool) {
 return _transfer(msg.sender, _to, _value);
 }

 function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
 require(_value <= allowances[_from][msg.sender]);
 allowances[_from][msg.sender] -= _value;
 return _transfer(_from, _to, _value);
 }

 function approve(address _spender, uint256 _value) public returns (bool) {
 allowances[msg.sender][_spender] = _value;
 Approval(msg.sender, _spender, _value);
 return true;
 }

 function burn(uint256 _value) public returns (bool) {
 require(balances[msg.sender] >= _value);
 balances[msg.sender] -= _value;
 totalSupply -= _value;
 Burn(msg.sender, _value);
 return true;
 }

 function burnFrom(address _from, uint256 _value) public returns (bool) {
 require(balances[_from] >= _value);
 require(_value <= allowances[_from][msg.sender]);
 balances[_from] -= _value;
 allowances[_from][msg.sender] -= _value;
 totalSupply -= _value;
 Burn(_from, _value);
 return true;
 }

 function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
 require(allowances[msg.sender][_spender] + _addedValue > allowances[msg.sender][_spender]);
 allowances[msg.sender][_spender] += _addedValue;
 Approval(msg.sender, _spender, allowances[msg.sender][_spender]);
 return true;
 }

 function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
 uint oldValue = allowances[msg.sender][_spender];
 if (_subtractedValue > oldValue) {
 allowances[msg.sender][_spender] = 0;
 } else {
 allowances[msg.sender][_spender] = oldValue - _subtractedValue;
 }
 Approval(msg.sender, _spender, allowances[msg.sender][_spender]);
 return true;
 }

}