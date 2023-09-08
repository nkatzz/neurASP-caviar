
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

	orientation(wk2gt_2,140,980). coords(wk2gt_2,97,65,980).  visible(wk2gt_2,980). happensAt(inactive(wk2gt_2),980). orientation(wk2gt_5,150,980). coords(wk2gt_5,94,108,980).  visible(wk2gt_5,980). happensAt(walking(wk2gt_5),980).
orientation(wk2gt_2,140,981). coords(wk2gt_2,97,65,981).  visible(wk2gt_2,981). happensAt(inactive(wk2gt_2),981). orientation(wk2gt_5,150,981). coords(wk2gt_5,94,108,981).  visible(wk2gt_5,981). happensAt(walking(wk2gt_5),981).
orientation(wk2gt_2,140,982). coords(wk2gt_2,97,65,982).  visible(wk2gt_2,982). happensAt(inactive(wk2gt_2),982). orientation(wk2gt_5,150,982). coords(wk2gt_5,95,108,982).  visible(wk2gt_5,982). happensAt(walking(wk2gt_5),982).
orientation(wk2gt_2,140,983). coords(wk2gt_2,97,65,983).  visible(wk2gt_2,983). happensAt(inactive(wk2gt_2),983). orientation(wk2gt_5,150,983). coords(wk2gt_5,95,108,983).  visible(wk2gt_5,983). happensAt(walking(wk2gt_5),983).
orientation(wk2gt_2,140,984). coords(wk2gt_2,97,65,984).  visible(wk2gt_2,984). happensAt(inactive(wk2gt_2),984). orientation(wk2gt_5,150,984). coords(wk2gt_5,95,109,984).  visible(wk2gt_5,984). happensAt(walking(wk2gt_5),984).
orientation(wk2gt_2,140,985). coords(wk2gt_2,97,65,985).  visible(wk2gt_2,985). happensAt(inactive(wk2gt_2),985). orientation(wk2gt_5,150,985). coords(wk2gt_5,95,110,985).  visible(wk2gt_5,985). happensAt(walking(wk2gt_5),985).
orientation(wk2gt_2,140,986). coords(wk2gt_2,97,65,986).  visible(wk2gt_2,986). happensAt(inactive(wk2gt_2),986). orientation(wk2gt_5,150,986). coords(wk2gt_5,95,110,986).  visible(wk2gt_5,986). happensAt(walking(wk2gt_5),986).
orientation(wk2gt_2,140,987). coords(wk2gt_2,97,65,987).  visible(wk2gt_2,987). happensAt(inactive(wk2gt_2),987). orientation(wk2gt_5,150,987). coords(wk2gt_5,95,112,987).  visible(wk2gt_5,987). happensAt(walking(wk2gt_5),987).
orientation(wk2gt_2,140,988). coords(wk2gt_2,97,65,988).  visible(wk2gt_2,988). happensAt(inactive(wk2gt_2),988). orientation(wk2gt_5,150,988). coords(wk2gt_5,97,113,988).  visible(wk2gt_5,988). happensAt(walking(wk2gt_5),988).
orientation(wk2gt_2,140,989). coords(wk2gt_2,97,65,989).  visible(wk2gt_2,989). happensAt(inactive(wk2gt_2),989). orientation(wk2gt_5,150,989). coords(wk2gt_5,98,113,989).  visible(wk2gt_5,989). happensAt(walking(wk2gt_5),989).
orientation(wk2gt_2,140,990). coords(wk2gt_2,97,65,990).  visible(wk2gt_2,990). happensAt(inactive(wk2gt_2),990). orientation(wk2gt_5,150,990). coords(wk2gt_5,99,114,990).  visible(wk2gt_5,990). happensAt(walking(wk2gt_5),990).
orientation(wk2gt_2,140,991). coords(wk2gt_2,97,65,991).  visible(wk2gt_2,991). happensAt(inactive(wk2gt_2),991). orientation(wk2gt_5,150,991). coords(wk2gt_5,101,114,991).  visible(wk2gt_5,991). happensAt(walking(wk2gt_5),991).
orientation(wk2gt_2,140,992). coords(wk2gt_2,97,65,992).  visible(wk2gt_2,992). happensAt(inactive(wk2gt_2),992). orientation(wk2gt_5,150,992). coords(wk2gt_5,101,114,992).  visible(wk2gt_5,992). happensAt(walking(wk2gt_5),992).
orientation(wk2gt_2,140,993). coords(wk2gt_2,97,65,993).  visible(wk2gt_2,993). happensAt(inactive(wk2gt_2),993). orientation(wk2gt_5,150,993). coords(wk2gt_5,102,115,993).  visible(wk2gt_5,993). happensAt(walking(wk2gt_5),993).
orientation(wk2gt_2,140,994). coords(wk2gt_2,97,65,994).  visible(wk2gt_2,994). happensAt(inactive(wk2gt_2),994). orientation(wk2gt_5,150,994). coords(wk2gt_5,102,115,994).  visible(wk2gt_5,994). happensAt(walking(wk2gt_5),994).
orientation(wk2gt_2,140,995). coords(wk2gt_2,97,65,995).  visible(wk2gt_2,995). happensAt(inactive(wk2gt_2),995). orientation(wk2gt_5,150,995). coords(wk2gt_5,102,115,995).  visible(wk2gt_5,995). happensAt(walking(wk2gt_5),995).
orientation(wk2gt_2,140,996). coords(wk2gt_2,97,65,996).  visible(wk2gt_2,996). happensAt(inactive(wk2gt_2),996). orientation(wk2gt_5,150,996). coords(wk2gt_5,102,115,996).  visible(wk2gt_5,996). happensAt(walking(wk2gt_5),996).
orientation(wk2gt_2,140,997). coords(wk2gt_2,97,65,997).  visible(wk2gt_2,997). happensAt(inactive(wk2gt_2),997). orientation(wk2gt_5,150,997). coords(wk2gt_5,102,115,997).  visible(wk2gt_5,997). happensAt(walking(wk2gt_5),997).
orientation(wk2gt_2,140,998). coords(wk2gt_2,97,65,998).  visible(wk2gt_2,998). happensAt(inactive(wk2gt_2),998). orientation(wk2gt_5,150,998). coords(wk2gt_5,104,115,998).  visible(wk2gt_5,998). happensAt(walking(wk2gt_5),998).
orientation(wk2gt_2,140,999). coords(wk2gt_2,97,65,999).  visible(wk2gt_2,999). happensAt(inactive(wk2gt_2),999). orientation(wk2gt_5,150,999). coords(wk2gt_5,105,116,999).  visible(wk2gt_5,999). happensAt(walking(wk2gt_5),999).
orientation(wk2gt_2,140,1000). coords(wk2gt_2,97,65,1000).  visible(wk2gt_2,1000). happensAt(inactive(wk2gt_2),1000). orientation(wk2gt_5,150,1000). coords(wk2gt_5,107,116,1000).  visible(wk2gt_5,1000). happensAt(walking(wk2gt_5),1000).
orientation(wk2gt_2,140,1001). coords(wk2gt_2,97,65,1001).  visible(wk2gt_2,1001). happensAt(inactive(wk2gt_2),1001). orientation(wk2gt_5,150,1001). coords(wk2gt_5,108,116,1001).  visible(wk2gt_5,1001). happensAt(walking(wk2gt_5),1001).
orientation(wk2gt_2,140,1002). coords(wk2gt_2,97,65,1002).  visible(wk2gt_2,1002). happensAt(inactive(wk2gt_2),1002). orientation(wk2gt_5,150,1002). coords(wk2gt_5,109,116,1002).  visible(wk2gt_5,1002). happensAt(walking(wk2gt_5),1002).
orientation(wk2gt_2,140,1003). coords(wk2gt_2,97,65,1003).  visible(wk2gt_2,1003). happensAt(inactive(wk2gt_2),1003). orientation(wk2gt_5,150,1003). coords(wk2gt_5,111,116,1003).  visible(wk2gt_5,1003). happensAt(walking(wk2gt_5),1003).
orientation(wk2gt_2,140,1004). coords(wk2gt_2,97,65,1004).  visible(wk2gt_2,1004). happensAt(inactive(wk2gt_2),1004). orientation(wk2gt_5,150,1004). coords(wk2gt_5,112,116,1004).  visible(wk2gt_5,1004). happensAt(walking(wk2gt_5),1004).
#show holdsAt/2.