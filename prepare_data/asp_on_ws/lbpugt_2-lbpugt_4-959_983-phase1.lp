
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbpugt_2,122,959). coords(lbpugt_2,189,66,959).  visible(lbpugt_2,959). happensAt(walking(lbpugt_2),959). orientation(lbpugt_4,135,959). coords(lbpugt_4,219,34,959).  visible(lbpugt_4,959). happensAt(inactive(lbpugt_4),959).
orientation(lbpugt_2,122,960). coords(lbpugt_2,189,66,960).  visible(lbpugt_2,960). happensAt(walking(lbpugt_2),960). orientation(lbpugt_4,135,960). coords(lbpugt_4,219,34,960).  visible(lbpugt_4,960). happensAt(inactive(lbpugt_4),960).
orientation(lbpugt_2,122,961). coords(lbpugt_2,190,65,961).  visible(lbpugt_2,961). happensAt(walking(lbpugt_2),961). orientation(lbpugt_4,135,961). coords(lbpugt_4,219,34,961).  visible(lbpugt_4,961). happensAt(inactive(lbpugt_4),961).
orientation(lbpugt_2,122,962). coords(lbpugt_2,190,64,962).  visible(lbpugt_2,962). happensAt(walking(lbpugt_2),962). orientation(lbpugt_4,135,962). coords(lbpugt_4,219,34,962).  visible(lbpugt_4,962). happensAt(inactive(lbpugt_4),962).
orientation(lbpugt_2,118,963). coords(lbpugt_2,189,62,963).  visible(lbpugt_2,963). happensAt(walking(lbpugt_2),963). orientation(lbpugt_4,135,963). coords(lbpugt_4,219,34,963).  visible(lbpugt_4,963). happensAt(inactive(lbpugt_4),963).
orientation(lbpugt_2,118,964). coords(lbpugt_2,190,60,964).  visible(lbpugt_2,964). happensAt(walking(lbpugt_2),964). orientation(lbpugt_4,135,964). coords(lbpugt_4,219,34,964).  visible(lbpugt_4,964). happensAt(inactive(lbpugt_4),964).
orientation(lbpugt_2,118,965). coords(lbpugt_2,190,59,965).  visible(lbpugt_2,965). happensAt(walking(lbpugt_2),965). orientation(lbpugt_4,135,965). coords(lbpugt_4,219,34,965).  visible(lbpugt_4,965). happensAt(inactive(lbpugt_4),965).
orientation(lbpugt_2,114,966). coords(lbpugt_2,190,56,966).  visible(lbpugt_2,966). happensAt(walking(lbpugt_2),966). orientation(lbpugt_4,135,966). coords(lbpugt_4,219,34,966).  visible(lbpugt_4,966). happensAt(inactive(lbpugt_4),966).
orientation(lbpugt_2,114,967). coords(lbpugt_2,190,54,967).  visible(lbpugt_2,967). happensAt(walking(lbpugt_2),967). orientation(lbpugt_4,135,967). coords(lbpugt_4,219,34,967).  visible(lbpugt_4,967). happensAt(inactive(lbpugt_4),967).
orientation(lbpugt_2,121,968). coords(lbpugt_2,190,53,968).  visible(lbpugt_2,968). happensAt(walking(lbpugt_2),968). orientation(lbpugt_4,135,968). coords(lbpugt_4,219,34,968).  visible(lbpugt_4,968). happensAt(inactive(lbpugt_4),968).
orientation(lbpugt_2,121,969). coords(lbpugt_2,190,52,969).  visible(lbpugt_2,969). happensAt(walking(lbpugt_2),969). orientation(lbpugt_4,135,969). coords(lbpugt_4,219,34,969).  visible(lbpugt_4,969). happensAt(inactive(lbpugt_4),969).
orientation(lbpugt_2,121,970). coords(lbpugt_2,190,52,970).  visible(lbpugt_2,970). happensAt(walking(lbpugt_2),970). orientation(lbpugt_4,135,970). coords(lbpugt_4,219,34,970).  visible(lbpugt_4,970). happensAt(inactive(lbpugt_4),970).
orientation(lbpugt_2,121,971). coords(lbpugt_2,190,51,971).  visible(lbpugt_2,971). happensAt(walking(lbpugt_2),971). orientation(lbpugt_4,135,971). coords(lbpugt_4,219,34,971).  visible(lbpugt_4,971). happensAt(inactive(lbpugt_4),971).
orientation(lbpugt_2,121,972). coords(lbpugt_2,191,51,972).  visible(lbpugt_2,972). happensAt(walking(lbpugt_2),972). orientation(lbpugt_4,135,972). coords(lbpugt_4,219,34,972).  visible(lbpugt_4,972). happensAt(inactive(lbpugt_4),972).
orientation(lbpugt_2,121,973). coords(lbpugt_2,191,50,973).  visible(lbpugt_2,973). happensAt(walking(lbpugt_2),973). orientation(lbpugt_4,135,973). coords(lbpugt_4,219,34,973).  visible(lbpugt_4,973). happensAt(inactive(lbpugt_4),973).
orientation(lbpugt_2,121,974). coords(lbpugt_2,192,50,974).  visible(lbpugt_2,974). happensAt(walking(lbpugt_2),974). orientation(lbpugt_4,135,974). coords(lbpugt_4,219,34,974).  visible(lbpugt_4,974). happensAt(inactive(lbpugt_4),974).
orientation(lbpugt_2,121,975). coords(lbpugt_2,192,49,975).  visible(lbpugt_2,975). happensAt(walking(lbpugt_2),975). orientation(lbpugt_4,135,975). coords(lbpugt_4,219,34,975).  visible(lbpugt_4,975). happensAt(inactive(lbpugt_4),975).
orientation(lbpugt_2,121,976). coords(lbpugt_2,192,47,976).  visible(lbpugt_2,976). happensAt(walking(lbpugt_2),976). orientation(lbpugt_4,135,976). coords(lbpugt_4,219,34,976).  visible(lbpugt_4,976). happensAt(inactive(lbpugt_4),976).
orientation(lbpugt_2,128,977). coords(lbpugt_2,192,46,977).  visible(lbpugt_2,977). happensAt(walking(lbpugt_2),977). orientation(lbpugt_4,135,977). coords(lbpugt_4,219,34,977).  visible(lbpugt_4,977). happensAt(inactive(lbpugt_4),977).
orientation(lbpugt_2,128,978). coords(lbpugt_2,192,44,978).  visible(lbpugt_2,978). happensAt(walking(lbpugt_2),978). orientation(lbpugt_4,135,978). coords(lbpugt_4,219,34,978).  visible(lbpugt_4,978). happensAt(inactive(lbpugt_4),978).
orientation(lbpugt_2,128,979). coords(lbpugt_2,192,44,979).  visible(lbpugt_2,979). happensAt(walking(lbpugt_2),979). orientation(lbpugt_4,135,979). coords(lbpugt_4,219,34,979).  visible(lbpugt_4,979). happensAt(inactive(lbpugt_4),979).
orientation(lbpugt_2,128,980). coords(lbpugt_2,194,43,980).  visible(lbpugt_2,980). happensAt(walking(lbpugt_2),980). orientation(lbpugt_4,135,980). coords(lbpugt_4,219,34,980).  visible(lbpugt_4,980). happensAt(inactive(lbpugt_4),980).
orientation(lbpugt_2,128,981). coords(lbpugt_2,196,42,981).  visible(lbpugt_2,981). happensAt(walking(lbpugt_2),981). orientation(lbpugt_4,135,981). coords(lbpugt_4,219,34,981).  visible(lbpugt_4,981). happensAt(inactive(lbpugt_4),981).
orientation(lbpugt_2,128,982). coords(lbpugt_2,196,41,982).  visible(lbpugt_2,982). happensAt(walking(lbpugt_2),982). orientation(lbpugt_4,135,982). coords(lbpugt_4,219,34,982).  visible(lbpugt_4,982). happensAt(inactive(lbpugt_4),982).
orientation(lbpugt_2,128,983). coords(lbpugt_2,197,41,983).  visible(lbpugt_2,983). happensAt(walking(lbpugt_2),983). orientation(lbpugt_4,135,983). coords(lbpugt_4,219,34,983).  visible(lbpugt_4,983). happensAt(inactive(lbpugt_4),983).
#show holdsAt/2.