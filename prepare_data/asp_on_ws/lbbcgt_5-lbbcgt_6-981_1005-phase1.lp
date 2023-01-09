
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

	orientation(lbbcgt_5,154,981). coords(lbbcgt_5,188,129,981).  visible(lbbcgt_5,981). happensAt(walking(lbbcgt_5),981). orientation(lbbcgt_6,151,981). coords(lbbcgt_6,73,90,981).  visible(lbbcgt_6,981). happensAt(walking(lbbcgt_6),981).
orientation(lbbcgt_5,154,982). coords(lbbcgt_5,188,129,982).  visible(lbbcgt_5,982). happensAt(walking(lbbcgt_5),982). orientation(lbbcgt_6,151,982). coords(lbbcgt_6,72,90,982).  visible(lbbcgt_6,982). happensAt(walking(lbbcgt_6),982).
orientation(lbbcgt_5,154,983). coords(lbbcgt_5,188,128,983).  visible(lbbcgt_5,983). happensAt(walking(lbbcgt_5),983). orientation(lbbcgt_6,151,983). coords(lbbcgt_6,72,90,983).  visible(lbbcgt_6,983). happensAt(walking(lbbcgt_6),983).
orientation(lbbcgt_5,154,984). coords(lbbcgt_5,188,127,984).  visible(lbbcgt_5,984). happensAt(walking(lbbcgt_5),984). orientation(lbbcgt_6,151,984). coords(lbbcgt_6,71,90,984).  visible(lbbcgt_6,984). happensAt(walking(lbbcgt_6),984).
orientation(lbbcgt_5,154,985). coords(lbbcgt_5,187,126,985).  visible(lbbcgt_5,985). happensAt(walking(lbbcgt_5),985). orientation(lbbcgt_6,151,985). coords(lbbcgt_6,71,90,985).  visible(lbbcgt_6,985). happensAt(walking(lbbcgt_6),985).
orientation(lbbcgt_5,154,986). coords(lbbcgt_5,187,126,986).  visible(lbbcgt_5,986). happensAt(walking(lbbcgt_5),986). orientation(lbbcgt_6,151,986). coords(lbbcgt_6,70,90,986).  visible(lbbcgt_6,986). happensAt(walking(lbbcgt_6),986).
orientation(lbbcgt_5,154,987). coords(lbbcgt_5,186,125,987).  visible(lbbcgt_5,987). happensAt(walking(lbbcgt_5),987). orientation(lbbcgt_6,151,987). coords(lbbcgt_6,70,90,987).  visible(lbbcgt_6,987). happensAt(walking(lbbcgt_6),987).
orientation(lbbcgt_5,154,988). coords(lbbcgt_5,184,124,988).  visible(lbbcgt_5,988). happensAt(walking(lbbcgt_5),988). orientation(lbbcgt_6,151,988). coords(lbbcgt_6,70,90,988).  visible(lbbcgt_6,988). happensAt(walking(lbbcgt_6),988).
orientation(lbbcgt_5,154,989). coords(lbbcgt_5,183,122,989).  visible(lbbcgt_5,989). happensAt(walking(lbbcgt_5),989). orientation(lbbcgt_6,151,989). coords(lbbcgt_6,70,91,989).  visible(lbbcgt_6,989). happensAt(walking(lbbcgt_6),989).
orientation(lbbcgt_5,154,990). coords(lbbcgt_5,182,121,990).  visible(lbbcgt_5,990). happensAt(walking(lbbcgt_5),990). orientation(lbbcgt_6,151,990). coords(lbbcgt_6,70,91,990).  visible(lbbcgt_6,990). happensAt(walking(lbbcgt_6),990).
orientation(lbbcgt_5,154,991). coords(lbbcgt_5,181,119,991).  visible(lbbcgt_5,991). happensAt(active(lbbcgt_5),991). orientation(lbbcgt_6,151,991). coords(lbbcgt_6,69,91,991).  visible(lbbcgt_6,991). happensAt(walking(lbbcgt_6),991).
orientation(lbbcgt_5,154,992). coords(lbbcgt_5,181,118,992).  visible(lbbcgt_5,992). happensAt(active(lbbcgt_5),992). orientation(lbbcgt_6,151,992). coords(lbbcgt_6,69,91,992).  visible(lbbcgt_6,992). happensAt(walking(lbbcgt_6),992).
orientation(lbbcgt_5,154,993). coords(lbbcgt_5,180,118,993).  visible(lbbcgt_5,993). happensAt(active(lbbcgt_5),993). orientation(lbbcgt_6,151,993). coords(lbbcgt_6,69,91,993).  visible(lbbcgt_6,993). happensAt(walking(lbbcgt_6),993).
orientation(lbbcgt_5,154,994). coords(lbbcgt_5,180,118,994).  visible(lbbcgt_5,994). happensAt(active(lbbcgt_5),994). orientation(lbbcgt_6,151,994). coords(lbbcgt_6,69,91,994).  visible(lbbcgt_6,994). happensAt(walking(lbbcgt_6),994).
orientation(lbbcgt_5,154,995). coords(lbbcgt_5,180,118,995).  visible(lbbcgt_5,995). happensAt(active(lbbcgt_5),995). orientation(lbbcgt_6,151,995). coords(lbbcgt_6,68,91,995).  visible(lbbcgt_6,995). happensAt(walking(lbbcgt_6),995).
orientation(lbbcgt_5,154,996). coords(lbbcgt_5,180,117,996).  visible(lbbcgt_5,996). happensAt(active(lbbcgt_5),996). orientation(lbbcgt_6,151,996). coords(lbbcgt_6,67,91,996).  visible(lbbcgt_6,996). happensAt(walking(lbbcgt_6),996).
orientation(lbbcgt_5,154,997). coords(lbbcgt_5,179,117,997).  visible(lbbcgt_5,997). happensAt(active(lbbcgt_5),997). orientation(lbbcgt_6,151,997). coords(lbbcgt_6,67,91,997).  visible(lbbcgt_6,997). happensAt(walking(lbbcgt_6),997).
orientation(lbbcgt_5,154,998). coords(lbbcgt_5,178,117,998).  visible(lbbcgt_5,998). happensAt(active(lbbcgt_5),998). orientation(lbbcgt_6,151,998). coords(lbbcgt_6,66,92,998).  visible(lbbcgt_6,998). happensAt(walking(lbbcgt_6),998).
orientation(lbbcgt_5,154,999). coords(lbbcgt_5,179,117,999).  visible(lbbcgt_5,999). happensAt(active(lbbcgt_5),999). orientation(lbbcgt_6,151,999). coords(lbbcgt_6,65,92,999).  visible(lbbcgt_6,999). happensAt(walking(lbbcgt_6),999).
orientation(lbbcgt_5,154,1000). coords(lbbcgt_5,180,117,1000).  visible(lbbcgt_5,1000). happensAt(active(lbbcgt_5),1000). orientation(lbbcgt_6,151,1000). coords(lbbcgt_6,65,93,1000).  visible(lbbcgt_6,1000). happensAt(walking(lbbcgt_6),1000).
orientation(lbbcgt_5,154,1001). coords(lbbcgt_5,180,117,1001).  visible(lbbcgt_5,1001). happensAt(active(lbbcgt_5),1001). orientation(lbbcgt_6,151,1001). coords(lbbcgt_6,65,93,1001).  visible(lbbcgt_6,1001). happensAt(walking(lbbcgt_6),1001).
orientation(lbbcgt_5,154,1002). coords(lbbcgt_5,180,117,1002).  visible(lbbcgt_5,1002). happensAt(active(lbbcgt_5),1002). orientation(lbbcgt_6,151,1002). coords(lbbcgt_6,65,94,1002).  visible(lbbcgt_6,1002). happensAt(walking(lbbcgt_6),1002).
orientation(lbbcgt_5,154,1003). coords(lbbcgt_5,179,117,1003).  visible(lbbcgt_5,1003). happensAt(active(lbbcgt_5),1003). orientation(lbbcgt_6,151,1003). coords(lbbcgt_6,65,94,1003).  visible(lbbcgt_6,1003). happensAt(walking(lbbcgt_6),1003).
orientation(lbbcgt_5,154,1004). coords(lbbcgt_5,179,117,1004).  visible(lbbcgt_5,1004). happensAt(active(lbbcgt_5),1004). orientation(lbbcgt_6,151,1004). coords(lbbcgt_6,65,94,1004).  visible(lbbcgt_6,1004). happensAt(walking(lbbcgt_6),1004).
orientation(lbbcgt_5,154,1005). coords(lbbcgt_5,179,117,1005).  visible(lbbcgt_5,1005). happensAt(active(lbbcgt_5),1005). orientation(lbbcgt_6,151,1005). coords(lbbcgt_6,64,94,1005).  visible(lbbcgt_6,1005). happensAt(walking(lbbcgt_6),1005).
#show holdsAt/2.