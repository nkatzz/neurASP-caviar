
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

	orientation(lbpugt_2,128,983). coords(lbpugt_2,197,41,983).  visible(lbpugt_2,983). happensAt(walking(lbpugt_2),983). orientation(lbpugt_4,135,983). coords(lbpugt_4,219,34,983).  visible(lbpugt_4,983). happensAt(inactive(lbpugt_4),983).
orientation(lbpugt_2,121,984). coords(lbpugt_2,198,40,984).  visible(lbpugt_2,984). happensAt(walking(lbpugt_2),984). orientation(lbpugt_4,135,984). coords(lbpugt_4,219,34,984).  visible(lbpugt_4,984). happensAt(inactive(lbpugt_4),984).
orientation(lbpugt_2,123,985). coords(lbpugt_2,198,39,985).  visible(lbpugt_2,985). happensAt(walking(lbpugt_2),985). orientation(lbpugt_4,135,985). coords(lbpugt_4,219,34,985).  visible(lbpugt_4,985). happensAt(inactive(lbpugt_4),985).
orientation(lbpugt_2,119,986). coords(lbpugt_2,199,39,986).  visible(lbpugt_2,986). happensAt(walking(lbpugt_2),986). orientation(lbpugt_4,135,986). coords(lbpugt_4,219,34,986).  visible(lbpugt_4,986). happensAt(inactive(lbpugt_4),986).
orientation(lbpugt_2,119,987). coords(lbpugt_2,199,39,987).  visible(lbpugt_2,987). happensAt(walking(lbpugt_2),987). orientation(lbpugt_4,135,987). coords(lbpugt_4,219,34,987).  visible(lbpugt_4,987). happensAt(inactive(lbpugt_4),987).
orientation(lbpugt_2,115,988). coords(lbpugt_2,200,38,988).  visible(lbpugt_2,988). happensAt(walking(lbpugt_2),988). orientation(lbpugt_4,135,988). coords(lbpugt_4,219,34,988).  visible(lbpugt_4,988). happensAt(inactive(lbpugt_4),988).
orientation(lbpugt_2,111,989). coords(lbpugt_2,202,38,989).  visible(lbpugt_2,989). happensAt(walking(lbpugt_2),989). orientation(lbpugt_4,135,989). coords(lbpugt_4,219,34,989).  visible(lbpugt_4,989). happensAt(inactive(lbpugt_4),989).
orientation(lbpugt_2,105,990). coords(lbpugt_2,202,38,990).  visible(lbpugt_2,990). happensAt(walking(lbpugt_2),990). orientation(lbpugt_4,135,990). coords(lbpugt_4,219,34,990).  visible(lbpugt_4,990). happensAt(inactive(lbpugt_4),990).
orientation(lbpugt_2,102,991). coords(lbpugt_2,202,38,991).  visible(lbpugt_2,991). happensAt(walking(lbpugt_2),991). orientation(lbpugt_4,135,991). coords(lbpugt_4,219,34,991).  visible(lbpugt_4,991). happensAt(inactive(lbpugt_4),991).
orientation(lbpugt_2,96,992). coords(lbpugt_2,203,38,992).  visible(lbpugt_2,992). happensAt(walking(lbpugt_2),992). orientation(lbpugt_4,135,992). coords(lbpugt_4,219,34,992).  visible(lbpugt_4,992). happensAt(inactive(lbpugt_4),992).
orientation(lbpugt_2,102,993). coords(lbpugt_2,203,38,993).  visible(lbpugt_2,993). happensAt(active(lbpugt_2),993). orientation(lbpugt_4,135,993). coords(lbpugt_4,219,34,993).  visible(lbpugt_4,993). happensAt(inactive(lbpugt_4),993).
orientation(lbpugt_2,92,994). coords(lbpugt_2,204,38,994).  visible(lbpugt_2,994). happensAt(active(lbpugt_2),994). orientation(lbpugt_4,135,994). coords(lbpugt_4,219,34,994).  visible(lbpugt_4,994). happensAt(inactive(lbpugt_4),994).
orientation(lbpugt_2,92,995). coords(lbpugt_2,204,38,995).  visible(lbpugt_2,995). happensAt(active(lbpugt_2),995). orientation(lbpugt_4,135,995). coords(lbpugt_4,219,34,995).  visible(lbpugt_4,995). happensAt(inactive(lbpugt_4),995).
orientation(lbpugt_2,92,996). coords(lbpugt_2,204,38,996).  visible(lbpugt_2,996). happensAt(active(lbpugt_2),996). orientation(lbpugt_4,135,996). coords(lbpugt_4,219,34,996).  visible(lbpugt_4,996). happensAt(inactive(lbpugt_4),996).
orientation(lbpugt_2,92,997). coords(lbpugt_2,204,38,997).  visible(lbpugt_2,997). happensAt(active(lbpugt_2),997). orientation(lbpugt_4,135,997). coords(lbpugt_4,219,34,997).  visible(lbpugt_4,997). happensAt(inactive(lbpugt_4),997).
orientation(lbpugt_2,92,998). coords(lbpugt_2,204,38,998).  visible(lbpugt_2,998). happensAt(active(lbpugt_2),998). orientation(lbpugt_4,135,998). coords(lbpugt_4,219,34,998).  visible(lbpugt_4,998). happensAt(inactive(lbpugt_4),998).
orientation(lbpugt_2,92,999). coords(lbpugt_2,204,38,999).  visible(lbpugt_2,999). happensAt(active(lbpugt_2),999). orientation(lbpugt_4,135,999). coords(lbpugt_4,219,34,999).  visible(lbpugt_4,999). happensAt(inactive(lbpugt_4),999).
orientation(lbpugt_2,92,1000). coords(lbpugt_2,204,38,1000).  visible(lbpugt_2,1000). happensAt(active(lbpugt_2),1000). orientation(lbpugt_4,135,1000). coords(lbpugt_4,219,34,1000).  visible(lbpugt_4,1000). happensAt(inactive(lbpugt_4),1000).
orientation(lbpugt_2,92,1001). coords(lbpugt_2,204,37,1001).  visible(lbpugt_2,1001). happensAt(active(lbpugt_2),1001). orientation(lbpugt_4,135,1001). coords(lbpugt_4,219,34,1001).  visible(lbpugt_4,1001). happensAt(inactive(lbpugt_4),1001).
orientation(lbpugt_2,95,1002). coords(lbpugt_2,204,37,1002).  visible(lbpugt_2,1002). happensAt(active(lbpugt_2),1002). orientation(lbpugt_4,135,1002). coords(lbpugt_4,219,34,1002).  visible(lbpugt_4,1002). happensAt(inactive(lbpugt_4),1002).
orientation(lbpugt_2,97,1003). coords(lbpugt_2,204,37,1003).  visible(lbpugt_2,1003). happensAt(active(lbpugt_2),1003). orientation(lbpugt_4,135,1003). coords(lbpugt_4,219,34,1003).  visible(lbpugt_4,1003). happensAt(inactive(lbpugt_4),1003).
orientation(lbpugt_2,102,1004). coords(lbpugt_2,204,37,1004).  visible(lbpugt_2,1004). happensAt(active(lbpugt_2),1004). orientation(lbpugt_4,135,1004). coords(lbpugt_4,219,34,1004).  visible(lbpugt_4,1004). happensAt(inactive(lbpugt_4),1004).
orientation(lbpugt_2,105,1005). coords(lbpugt_2,204,37,1005).  visible(lbpugt_2,1005). happensAt(active(lbpugt_2),1005). orientation(lbpugt_4,135,1005). coords(lbpugt_4,219,34,1005).  visible(lbpugt_4,1005). happensAt(inactive(lbpugt_4),1005).
orientation(lbpugt_2,105,1006). coords(lbpugt_2,204,37,1006).  visible(lbpugt_2,1006). happensAt(active(lbpugt_2),1006). orientation(lbpugt_4,135,1006). coords(lbpugt_4,219,34,1006).  visible(lbpugt_4,1006). happensAt(inactive(lbpugt_4),1006).
orientation(lbpugt_2,106,1007). coords(lbpugt_2,204,36,1007).  visible(lbpugt_2,1007). happensAt(active(lbpugt_2),1007). orientation(lbpugt_4,135,1007). coords(lbpugt_4,218,34,1007).  visible(lbpugt_4,1007). happensAt(inactive(lbpugt_4),1007).
#show holdsAt/2.