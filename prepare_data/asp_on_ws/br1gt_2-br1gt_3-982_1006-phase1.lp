
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

	orientation(br1gt_2,166,982). coords(br1gt_2,105,208,982).  visible(br1gt_2,982). happensAt(walking(br1gt_2),982). orientation(br1gt_3,122,982). coords(br1gt_3,292,94,982).  visible(br1gt_3,982). happensAt(walking(br1gt_3),982).
orientation(br1gt_2,166,983). coords(br1gt_2,106,208,983).  visible(br1gt_2,983). happensAt(walking(br1gt_2),983). orientation(br1gt_3,122,983). coords(br1gt_3,290,94,983).  visible(br1gt_3,983). happensAt(walking(br1gt_3),983).
orientation(br1gt_2,166,984). coords(br1gt_2,106,208,984).  visible(br1gt_2,984). happensAt(walking(br1gt_2),984). orientation(br1gt_3,122,984). coords(br1gt_3,287,95,984).  visible(br1gt_3,984). happensAt(walking(br1gt_3),984).
orientation(br1gt_2,166,985). coords(br1gt_2,107,208,985).  visible(br1gt_2,985). happensAt(walking(br1gt_2),985). orientation(br1gt_3,122,985). coords(br1gt_3,284,95,985).  visible(br1gt_3,985). happensAt(walking(br1gt_3),985).
orientation(br1gt_2,166,986). coords(br1gt_2,107,208,986).  visible(br1gt_2,986). happensAt(walking(br1gt_2),986). orientation(br1gt_3,118,986). coords(br1gt_3,283,96,986).  visible(br1gt_3,986). happensAt(walking(br1gt_3),986).
orientation(br1gt_2,166,987). coords(br1gt_2,107,208,987).  visible(br1gt_2,987). happensAt(walking(br1gt_2),987). orientation(br1gt_3,118,987). coords(br1gt_3,282,96,987).  visible(br1gt_3,987). happensAt(walking(br1gt_3),987).
orientation(br1gt_2,166,988). coords(br1gt_2,107,208,988).  visible(br1gt_2,988). happensAt(walking(br1gt_2),988). orientation(br1gt_3,118,988). coords(br1gt_3,281,96,988).  visible(br1gt_3,988). happensAt(walking(br1gt_3),988).
orientation(br1gt_2,166,989). coords(br1gt_2,108,208,989).  visible(br1gt_2,989). happensAt(walking(br1gt_2),989). orientation(br1gt_3,118,989). coords(br1gt_3,281,96,989).  visible(br1gt_3,989). happensAt(walking(br1gt_3),989).
orientation(br1gt_2,166,990). coords(br1gt_2,108,208,990).  visible(br1gt_2,990). happensAt(walking(br1gt_2),990). orientation(br1gt_3,118,990). coords(br1gt_3,280,96,990).  visible(br1gt_3,990). happensAt(walking(br1gt_3),990).
orientation(br1gt_2,166,991). coords(br1gt_2,108,208,991).  visible(br1gt_2,991). happensAt(walking(br1gt_2),991). orientation(br1gt_3,118,991). coords(br1gt_3,279,95,991).  visible(br1gt_3,991). happensAt(walking(br1gt_3),991).
orientation(br1gt_2,174,992). coords(br1gt_2,108,208,992).  visible(br1gt_2,992). happensAt(walking(br1gt_2),992). orientation(br1gt_3,123,992). coords(br1gt_3,279,95,992).  visible(br1gt_3,992). happensAt(walking(br1gt_3),992).
orientation(br1gt_2,174,993). coords(br1gt_2,108,208,993).  visible(br1gt_2,993). happensAt(walking(br1gt_2),993). orientation(br1gt_3,123,993). coords(br1gt_3,279,95,993).  visible(br1gt_3,993). happensAt(walking(br1gt_3),993).
orientation(br1gt_2,174,994). coords(br1gt_2,108,209,994).  visible(br1gt_2,994). happensAt(walking(br1gt_2),994). orientation(br1gt_3,123,994). coords(br1gt_3,278,95,994).  visible(br1gt_3,994). happensAt(walking(br1gt_3),994).
orientation(br1gt_2,174,995). coords(br1gt_2,109,209,995).  visible(br1gt_2,995). happensAt(walking(br1gt_2),995). orientation(br1gt_3,123,995). coords(br1gt_3,276,95,995).  visible(br1gt_3,995). happensAt(walking(br1gt_3),995).
orientation(br1gt_2,174,996). coords(br1gt_2,109,209,996).  visible(br1gt_2,996). happensAt(walking(br1gt_2),996). orientation(br1gt_3,123,996). coords(br1gt_3,275,95,996).  visible(br1gt_3,996). happensAt(walking(br1gt_3),996).
orientation(br1gt_2,174,997). coords(br1gt_2,109,210,997).  visible(br1gt_2,997). happensAt(walking(br1gt_2),997). orientation(br1gt_3,123,997). coords(br1gt_3,273,95,997).  visible(br1gt_3,997). happensAt(walking(br1gt_3),997).
orientation(br1gt_2,174,998). coords(br1gt_2,109,210,998).  visible(br1gt_2,998). happensAt(walking(br1gt_2),998). orientation(br1gt_3,123,998). coords(br1gt_3,270,96,998).  visible(br1gt_3,998). happensAt(walking(br1gt_3),998).
orientation(br1gt_2,174,999). coords(br1gt_2,109,211,999).  visible(br1gt_2,999). happensAt(walking(br1gt_2),999). orientation(br1gt_3,123,999). coords(br1gt_3,268,96,999).  visible(br1gt_3,999). happensAt(walking(br1gt_3),999).
orientation(br1gt_2,174,1000). coords(br1gt_2,109,211,1000).  visible(br1gt_2,1000). happensAt(walking(br1gt_2),1000). orientation(br1gt_3,123,1000). coords(br1gt_3,266,96,1000).  visible(br1gt_3,1000). happensAt(walking(br1gt_3),1000).
orientation(br1gt_2,174,1001). coords(br1gt_2,109,212,1001).  visible(br1gt_2,1001). happensAt(walking(br1gt_2),1001). orientation(br1gt_3,123,1001). coords(br1gt_3,263,96,1001).  visible(br1gt_3,1001). happensAt(walking(br1gt_3),1001).
orientation(br1gt_2,174,1002). coords(br1gt_2,110,212,1002).  visible(br1gt_2,1002). happensAt(walking(br1gt_2),1002). orientation(br1gt_3,123,1002). coords(br1gt_3,260,96,1002).  visible(br1gt_3,1002). happensAt(walking(br1gt_3),1002).
orientation(br1gt_2,174,1003). coords(br1gt_2,110,212,1003).  visible(br1gt_2,1003). happensAt(walking(br1gt_2),1003). orientation(br1gt_3,123,1003). coords(br1gt_3,258,97,1003).  visible(br1gt_3,1003). happensAt(walking(br1gt_3),1003).
orientation(br1gt_2,174,1004). coords(br1gt_2,111,213,1004).  visible(br1gt_2,1004). happensAt(walking(br1gt_2),1004). orientation(br1gt_3,123,1004). coords(br1gt_3,258,97,1004).  visible(br1gt_3,1004). happensAt(walking(br1gt_3),1004).
orientation(br1gt_2,174,1005). coords(br1gt_2,110,214,1005).  visible(br1gt_2,1005). happensAt(walking(br1gt_2),1005). orientation(br1gt_3,128,1005). coords(br1gt_3,257,97,1005).  visible(br1gt_3,1005). happensAt(walking(br1gt_3),1005).
orientation(br1gt_2,174,1006). coords(br1gt_2,110,214,1006).  visible(br1gt_2,1006). happensAt(walking(br1gt_2),1006). orientation(br1gt_3,128,1006). coords(br1gt_3,256,97,1006).  visible(br1gt_3,1006). happensAt(walking(br1gt_3),1006).
#show holdsAt/2.