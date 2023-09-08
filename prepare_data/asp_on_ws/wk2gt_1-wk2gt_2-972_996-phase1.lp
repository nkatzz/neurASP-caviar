
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

	orientation(wk2gt_1,133,972). coords(wk2gt_1,128,44,972).  visible(wk2gt_1,972). happensAt(walking(wk2gt_1),972). orientation(wk2gt_2,137,972). coords(wk2gt_2,97,65,972).  visible(wk2gt_2,972). happensAt(active(wk2gt_2),972).
orientation(wk2gt_1,133,973). coords(wk2gt_1,128,44,973).  visible(wk2gt_1,973). happensAt(walking(wk2gt_1),973). orientation(wk2gt_2,137,973). coords(wk2gt_2,97,65,973).  visible(wk2gt_2,973). happensAt(active(wk2gt_2),973).
orientation(wk2gt_1,133,974). coords(wk2gt_1,128,43,974).  visible(wk2gt_1,974). happensAt(walking(wk2gt_1),974). orientation(wk2gt_2,137,974). coords(wk2gt_2,97,65,974).  visible(wk2gt_2,974). happensAt(active(wk2gt_2),974).
orientation(wk2gt_1,133,975). coords(wk2gt_1,128,43,975).  visible(wk2gt_1,975). happensAt(walking(wk2gt_1),975). orientation(wk2gt_2,137,975). coords(wk2gt_2,97,65,975).  visible(wk2gt_2,975). happensAt(active(wk2gt_2),975).
orientation(wk2gt_1,133,976). coords(wk2gt_1,129,43,976).  visible(wk2gt_1,976). happensAt(walking(wk2gt_1),976). orientation(wk2gt_2,137,976). coords(wk2gt_2,97,65,976).  visible(wk2gt_2,976). happensAt(inactive(wk2gt_2),976).
orientation(wk2gt_1,133,977). coords(wk2gt_1,129,43,977).  visible(wk2gt_1,977). happensAt(walking(wk2gt_1),977). orientation(wk2gt_2,137,977). coords(wk2gt_2,97,65,977).  visible(wk2gt_2,977). happensAt(inactive(wk2gt_2),977).
orientation(wk2gt_1,133,978). coords(wk2gt_1,129,43,978).  visible(wk2gt_1,978). happensAt(walking(wk2gt_1),978). orientation(wk2gt_2,137,978). coords(wk2gt_2,97,65,978).  visible(wk2gt_2,978). happensAt(inactive(wk2gt_2),978).
orientation(wk2gt_1,133,979). coords(wk2gt_1,129,43,979).  visible(wk2gt_1,979). happensAt(walking(wk2gt_1),979). orientation(wk2gt_2,140,979). coords(wk2gt_2,97,65,979).  visible(wk2gt_2,979). happensAt(inactive(wk2gt_2),979).
orientation(wk2gt_1,133,980). coords(wk2gt_1,130,43,980).  visible(wk2gt_1,980). happensAt(walking(wk2gt_1),980). orientation(wk2gt_2,140,980). coords(wk2gt_2,97,65,980).  visible(wk2gt_2,980). happensAt(inactive(wk2gt_2),980).
orientation(wk2gt_1,133,981). coords(wk2gt_1,130,43,981).  visible(wk2gt_1,981). happensAt(walking(wk2gt_1),981). orientation(wk2gt_2,140,981). coords(wk2gt_2,97,65,981).  visible(wk2gt_2,981). happensAt(inactive(wk2gt_2),981).
orientation(wk2gt_1,133,982). coords(wk2gt_1,130,43,982).  visible(wk2gt_1,982). happensAt(walking(wk2gt_1),982). orientation(wk2gt_2,140,982). coords(wk2gt_2,97,65,982).  visible(wk2gt_2,982). happensAt(inactive(wk2gt_2),982).
orientation(wk2gt_1,133,983). coords(wk2gt_1,130,43,983).  visible(wk2gt_1,983). happensAt(walking(wk2gt_1),983). orientation(wk2gt_2,140,983). coords(wk2gt_2,97,65,983).  visible(wk2gt_2,983). happensAt(inactive(wk2gt_2),983).
orientation(wk2gt_1,133,984). coords(wk2gt_1,130,43,984).  visible(wk2gt_1,984). happensAt(walking(wk2gt_1),984). orientation(wk2gt_2,140,984). coords(wk2gt_2,97,65,984).  visible(wk2gt_2,984). happensAt(inactive(wk2gt_2),984).
orientation(wk2gt_1,133,985). coords(wk2gt_1,130,43,985).  visible(wk2gt_1,985). happensAt(walking(wk2gt_1),985). orientation(wk2gt_2,140,985). coords(wk2gt_2,97,65,985).  visible(wk2gt_2,985). happensAt(inactive(wk2gt_2),985).
orientation(wk2gt_1,133,986). coords(wk2gt_1,130,43,986).  visible(wk2gt_1,986). happensAt(walking(wk2gt_1),986). orientation(wk2gt_2,140,986). coords(wk2gt_2,97,65,986).  visible(wk2gt_2,986). happensAt(inactive(wk2gt_2),986).
orientation(wk2gt_1,133,987). coords(wk2gt_1,130,42,987).  visible(wk2gt_1,987). happensAt(walking(wk2gt_1),987). orientation(wk2gt_2,140,987). coords(wk2gt_2,97,65,987).  visible(wk2gt_2,987). happensAt(inactive(wk2gt_2),987).
orientation(wk2gt_1,133,988). coords(wk2gt_1,130,43,988).  visible(wk2gt_1,988). happensAt(walking(wk2gt_1),988). orientation(wk2gt_2,140,988). coords(wk2gt_2,97,65,988).  visible(wk2gt_2,988). happensAt(inactive(wk2gt_2),988).
orientation(wk2gt_1,133,989). coords(wk2gt_1,131,42,989).  visible(wk2gt_1,989). happensAt(walking(wk2gt_1),989). orientation(wk2gt_2,140,989). coords(wk2gt_2,97,65,989).  visible(wk2gt_2,989). happensAt(inactive(wk2gt_2),989).
orientation(wk2gt_1,133,990). coords(wk2gt_1,131,42,990).  visible(wk2gt_1,990). happensAt(walking(wk2gt_1),990). orientation(wk2gt_2,140,990). coords(wk2gt_2,97,65,990).  visible(wk2gt_2,990). happensAt(inactive(wk2gt_2),990).
orientation(wk2gt_1,133,991). coords(wk2gt_1,132,42,991).  visible(wk2gt_1,991). happensAt(walking(wk2gt_1),991). orientation(wk2gt_2,140,991). coords(wk2gt_2,97,65,991).  visible(wk2gt_2,991). happensAt(inactive(wk2gt_2),991).
orientation(wk2gt_1,133,992). coords(wk2gt_1,132,42,992).  visible(wk2gt_1,992). happensAt(walking(wk2gt_1),992). orientation(wk2gt_2,140,992). coords(wk2gt_2,97,65,992).  visible(wk2gt_2,992). happensAt(inactive(wk2gt_2),992).
orientation(wk2gt_1,133,993). coords(wk2gt_1,132,41,993).  visible(wk2gt_1,993). happensAt(walking(wk2gt_1),993). orientation(wk2gt_2,140,993). coords(wk2gt_2,97,65,993).  visible(wk2gt_2,993). happensAt(inactive(wk2gt_2),993).
orientation(wk2gt_1,133,994). coords(wk2gt_1,133,41,994).  visible(wk2gt_1,994). happensAt(walking(wk2gt_1),994). orientation(wk2gt_2,140,994). coords(wk2gt_2,97,65,994).  visible(wk2gt_2,994). happensAt(inactive(wk2gt_2),994).
orientation(wk2gt_1,133,995). coords(wk2gt_1,133,41,995).  visible(wk2gt_1,995). happensAt(walking(wk2gt_1),995). orientation(wk2gt_2,140,995). coords(wk2gt_2,97,65,995).  visible(wk2gt_2,995). happensAt(inactive(wk2gt_2),995).
orientation(wk2gt_1,133,996). coords(wk2gt_1,133,41,996).  visible(wk2gt_1,996). happensAt(walking(wk2gt_1),996). orientation(wk2gt_2,140,996). coords(wk2gt_2,97,65,996).  visible(wk2gt_2,996). happensAt(inactive(wk2gt_2),996).
#show holdsAt/2.