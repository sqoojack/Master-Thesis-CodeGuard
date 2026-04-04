 contract USDYToken is PausableToken {
    uint public INITIAL_SUPPLY = 1500000000000;

    function USDYToken() public {
        name = "yether";
        symbol = "USDY";
        decimals = 2;
        totalSupply = INITIAL_SUPPLY;
        balances[msg.sender] = INITIAL_SUPPLY;
    }

    function transfer(address _to, uint256 _value) public whenNotPaused returns (bool) {
        return super.transfer(_to, _value);
    }

    function transferFrom(address _from, address _to, uint256 _value) public whenNotPaused returns (bool) {
        return super.transferFrom(_from, _to, _value);
    }

    function approve(address _spender, uint256 _value) public whenNotPaused returns (bool) {
        return super.approve(_spender, _value);
    }

    function increaseApproval(address _spender, uint _addedValue) public whenNotPaused returns (bool success) {
        return super.increaseApproval(_spender, _addedValue);
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public whenNotPaused returns (bool success) {
        return super.decreaseApproval(_spender, _subtractedValue);
    }

    function setname(string _name) onlyOwner public whenNotPaused returns (bool){
        name = _name;
        return true;
    }

    function setsymbol(string _symbol) onlyOwner public whenNotPaused returns (bool){
        symbol = _symbol;
        return true;
    }

    function setdecimals(uint _decimals) onlyOwner public whenNotPaused returns (bool){
        decimals = _decimals;
        return true;
    }

    function mintToken(address target, uint256 _value) onlyOwner public whenNotPaused returns (bool){
        require(target!= address(0));
        balances[target] = balances[target].add(_value);
        totalSupply = totalSupply.add(_value);
        Transfer(0, owner, _value);
        Transfer(owner, target, _value);
        return true;
    }
}