{
    deque <int>	U, L;
    for (uint i = 1; i < a.size(); ++i)
    {
	if (i >= w)
	{
	    maxval[i - w] = a [U.size() > 0 ? U.front() : i - 1] ;
	    minval[i - w] = a [L.size() > 0 ? L.front() : i - 1] ;
	}
	
	if (a[i] > a[i - 1])
	{
	    L.push_back(i - 1);
	    if (i == w + L.front())
		L.pop_front();
	    
	    while (U.size() > 0)
	    {
		if (a[i] <= a[U.back()])
		{
		    if (i ==  w + U.front())
			U.pop_front();
		    break;
		}

		U.pop_back();
	    }
	}
	else
	{
	    U.push_back(i - 1);
	    if (i == w + U.front())
		U.pop_front();

	    while (L.size() >0 )
	    {
		if (a[i] >= a[L.back()])
		{
		    if (i == w + L.front())
			L.pop_front();
		    break ;
		}
	    
		L.pop_back();
	    }
	}
    }

    maxval[a.size() - w] = a [U.size() > 0 ? U.front() : a.size() - 1];
    minval[a.size() - w] = a [L.size() > 0 ? L.front() : a.size() - 1];
}

