
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

	orientation(lbbcgt_3,163,965). coords(lbbcgt_3,35,199,965).  visible(lbbcgt_3,965). happensAt(walking(lbbcgt_3),965). orientation(lbbcgt_6,151,965). coords(lbbcgt_6,79,87,965).  visible(lbbcgt_6,965). happensAt(walking(lbbcgt_6),965).
orientation(lbbcgt_3,163,966). coords(lbbcgt_3,35,199,966).  visible(lbbcgt_3,966). happensAt(walking(lbbcgt_3),966). orientation(lbbcgt_6,151,966). coords(lbbcgt_6,79,86,966).  visible(lbbcgt_6,966). happensAt(walking(lbbcgt_6),966).
orientation(lbbcgt_3,163,967). coords(lbbcgt_3,36,199,967).  visible(lbbcgt_3,967). happensAt(walking(lbbcgt_3),967). orientation(lbbcgt_6,151,967). coords(lbbcgt_6,78,87,967).  visible(lbbcgt_6,967). happensAt(walking(lbbcgt_6),967).
orientation(lbbcgt_3,163,968). coords(lbbcgt_3,36,200,968).  visible(lbbcgt_3,968). happensAt(walking(lbbcgt_3),968). orientation(lbbcgt_6,151,968). coords(lbbcgt_6,78,87,968).  visible(lbbcgt_6,968). happensAt(walking(lbbcgt_6),968).
orientation(lbbcgt_3,163,969). coords(lbbcgt_3,36,200,969).  visible(lbbcgt_3,969). happensAt(walking(lbbcgt_3),969). orientation(lbbcgt_6,151,969). coords(lbbcgt_6,77,87,969).  visible(lbbcgt_6,969). happensAt(walking(lbbcgt_6),969).
orientation(lbbcgt_3,163,970). coords(lbbcgt_3,35,201,970).  visible(lbbcgt_3,970). happensAt(walking(lbbcgt_3),970). orientation(lbbcgt_6,151,970). coords(lbbcgt_6,76,87,970).  visible(lbbcgt_6,970). happensAt(walking(lbbcgt_6),970).
orientation(lbbcgt_3,163,971). coords(lbbcgt_3,35,201,971).  visible(lbbcgt_3,971). happensAt(walking(lbbcgt_3),971). orientation(lbbcgt_6,151,971). coords(lbbcgt_6,76,87,971).  visible(lbbcgt_6,971). happensAt(walking(lbbcgt_6),971).
orientation(lbbcgt_3,163,972). coords(lbbcgt_3,35,202,972).  visible(lbbcgt_3,972). happensAt(walking(lbbcgt_3),972). orientation(lbbcgt_6,151,972). coords(lbbcgt_6,75,88,972).  visible(lbbcgt_6,972). happensAt(walking(lbbcgt_6),972).
orientation(lbbcgt_3,163,973). coords(lbbcgt_3,35,203,973).  visible(lbbcgt_3,973). happensAt(walking(lbbcgt_3),973). orientation(lbbcgt_6,151,973). coords(lbbcgt_6,76,88,973).  visible(lbbcgt_6,973). happensAt(walking(lbbcgt_6),973).
orientation(lbbcgt_3,163,974). coords(lbbcgt_3,35,204,974).  visible(lbbcgt_3,974). happensAt(walking(lbbcgt_3),974). orientation(lbbcgt_6,151,974). coords(lbbcgt_6,75,88,974).  visible(lbbcgt_6,974). happensAt(walking(lbbcgt_6),974).
orientation(lbbcgt_3,163,975). coords(lbbcgt_3,35,205,975).  visible(lbbcgt_3,975). happensAt(walking(lbbcgt_3),975). orientation(lbbcgt_6,151,975). coords(lbbcgt_6,75,88,975).  visible(lbbcgt_6,975). happensAt(walking(lbbcgt_6),975).
orientation(lbbcgt_3,163,976). coords(lbbcgt_3,35,205,976).  visible(lbbcgt_3,976). happensAt(walking(lbbcgt_3),976). orientation(lbbcgt_6,151,976). coords(lbbcgt_6,75,88,976).  visible(lbbcgt_6,976). happensAt(walking(lbbcgt_6),976).
orientation(lbbcgt_3,163,977). coords(lbbcgt_3,35,206,977).  visible(lbbcgt_3,977). happensAt(walking(lbbcgt_3),977). orientation(lbbcgt_6,151,977). coords(lbbcgt_6,75,89,977).  visible(lbbcgt_6,977). happensAt(walking(lbbcgt_6),977).
orientation(lbbcgt_3,163,978). coords(lbbcgt_3,35,207,978).  visible(lbbcgt_3,978). happensAt(walking(lbbcgt_3),978). orientation(lbbcgt_6,151,978). coords(lbbcgt_6,75,89,978).  visible(lbbcgt_6,978). happensAt(walking(lbbcgt_6),978).
orientation(lbbcgt_3,163,979). coords(lbbcgt_3,35,207,979).  visible(lbbcgt_3,979). happensAt(walking(lbbcgt_3),979). orientation(lbbcgt_6,151,979). coords(lbbcgt_6,75,89,979).  visible(lbbcgt_6,979). happensAt(walking(lbbcgt_6),979).
orientation(lbbcgt_3,163,980). coords(lbbcgt_3,35,207,980).  visible(lbbcgt_3,980). happensAt(walking(lbbcgt_3),980). orientation(lbbcgt_6,151,980). coords(lbbcgt_6,74,89,980).  visible(lbbcgt_6,980). happensAt(walking(lbbcgt_6),980).
orientation(lbbcgt_3,163,981). coords(lbbcgt_3,31,207,981).  visible(lbbcgt_3,981). happensAt(walking(lbbcgt_3),981). orientation(lbbcgt_6,151,981). coords(lbbcgt_6,73,90,981).  visible(lbbcgt_6,981). happensAt(walking(lbbcgt_6),981).
orientation(lbbcgt_3,163,982). coords(lbbcgt_3,31,208,982).  visible(lbbcgt_3,982). happensAt(walking(lbbcgt_3),982). orientation(lbbcgt_6,151,982). coords(lbbcgt_6,72,90,982).  visible(lbbcgt_6,982). happensAt(walking(lbbcgt_6),982).
orientation(lbbcgt_3,163,983). coords(lbbcgt_3,30,208,983).  visible(lbbcgt_3,983). happensAt(walking(lbbcgt_3),983). orientation(lbbcgt_6,151,983). coords(lbbcgt_6,72,90,983).  visible(lbbcgt_6,983). happensAt(walking(lbbcgt_6),983).
orientation(lbbcgt_3,163,984). coords(lbbcgt_3,30,209,984).  visible(lbbcgt_3,984). happensAt(walking(lbbcgt_3),984). orientation(lbbcgt_6,151,984). coords(lbbcgt_6,71,90,984).  visible(lbbcgt_6,984). happensAt(walking(lbbcgt_6),984).
orientation(lbbcgt_3,163,985). coords(lbbcgt_3,29,209,985).  visible(lbbcgt_3,985). happensAt(walking(lbbcgt_3),985). orientation(lbbcgt_6,151,985). coords(lbbcgt_6,71,90,985).  visible(lbbcgt_6,985). happensAt(walking(lbbcgt_6),985).
orientation(lbbcgt_3,163,986). coords(lbbcgt_3,28,209,986).  visible(lbbcgt_3,986). happensAt(walking(lbbcgt_3),986). orientation(lbbcgt_6,151,986). coords(lbbcgt_6,70,90,986).  visible(lbbcgt_6,986). happensAt(walking(lbbcgt_6),986).
orientation(lbbcgt_3,163,987). coords(lbbcgt_3,28,209,987).  visible(lbbcgt_3,987). happensAt(walking(lbbcgt_3),987). orientation(lbbcgt_6,151,987). coords(lbbcgt_6,70,90,987).  visible(lbbcgt_6,987). happensAt(walking(lbbcgt_6),987).
orientation(lbbcgt_3,163,988). coords(lbbcgt_3,28,211,988).  visible(lbbcgt_3,988). happensAt(walking(lbbcgt_3),988). orientation(lbbcgt_6,151,988). coords(lbbcgt_6,70,90,988).  visible(lbbcgt_6,988). happensAt(walking(lbbcgt_6),988).
orientation(lbbcgt_3,163,989). coords(lbbcgt_3,28,211,989).  visible(lbbcgt_3,989). happensAt(walking(lbbcgt_3),989). orientation(lbbcgt_6,151,989). coords(lbbcgt_6,70,91,989).  visible(lbbcgt_6,989). happensAt(walking(lbbcgt_6),989).
#show holdsAt/2.