 contract STBToken is TokenERC20 {

    function _transfer(address _from, address _to, uint _value) internal returns (bool) {
        require(_to!= 0x0);
        require(balances[_from] >= _value);
        require(balances[_to] + _value > balances[_to]);

        require(_value >= 0);
        uint previousBalances = balances[_from].add(balances[_to]);
        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(_from, _to, _value);
        assert(balances[_from] + balances[_to] == previousBalances);

        return true;
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        return _transfer(msg.sender, _to, _value);
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(_to!= address(0));
        require(_value <= balances[_from]);
        require(_value > 0);

        balances[_from] = balances[_from].sub(_value);
        balances[_to] = balances[_to].add(_value);
        allowances[_from][msg.sender] = allowances[_from][msg.sender].sub(_value);
        Transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool) {
        require((_value == 0) || (allowances[msg.sender][_spender] == 0));
        allowances[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferForMultiAddresses(address[] _addresses, uint256[] _amounts) public returns (bool) {
        for (uint256 i = 0; i < _addresses.length; i++) {
            require(_addresses[i]!= address(0));
            require(_amounts[i] <= balances[msg.sender]);
            require(_amounts[i] > 0);

            balances[msg.sender] = balances[msg.sender].sub(_amounts[i]);
            balances[_addresses[i]] = balances[_addresses[i]].add(_amounts[i]);
            Transfer(msg.sender, _addresses[i], _amounts[i]);
        }
        return true;
    }

    function burn(uint256 _value) public returns(bool) {
        require(balances[msg.sender] >= _value);
        balances[msg.sender] = balances[msg.sender].sub(_value);
        totalSupply = totalSupply.sub(_value);
        Burn(msg.sender, _value);
        return true;
    }

    function burnFrom(address _from, uint256 _value) public returns(bool) {
        require(balances[_from] >= _value);
        require(_value <= allowances[_from][msg.sender]);
        balances[_from] = balances[_from].sub(_value);
        allowances[_from][msg.sender] = allowances[_from][msg.sender].sub(_value);
        totalSupply = totalSupply.sub(_value);
        Burn(_from, _value);
        return true;
    }

    function increaseApproval(address _spender, uint _addedValue) public returns (bool) {
        require(allowances[msg.sender][_spender].add(_addedValue) > allowances[msg.sender][_spender]);

        allowances[msg.sender][_spender] =allowances[msg.sender][_spender].add(_addedValue);
        Approval(msg.sender, _spender, allowances[msg.sender][_spender]);
        return true;
    }

    function decreaseApproval(address _spender, uint _subtractedValue) public returns (bool) {
        uint oldValue