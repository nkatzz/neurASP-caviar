
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

	orientation(mws1gt_4,0,168). coords(mws1gt_4,29,122,168).  visible(mws1gt_4,168). happensAt(walking(mws1gt_4),168). orientation(mws1gt_6,146,168). coords(mws1gt_6,73,65,168).  visible(mws1gt_6,168). happensAt(walking(mws1gt_6),168).
orientation(mws1gt_4,0,169). coords(mws1gt_4,29,122,169).  visible(mws1gt_4,169). happensAt(walking(mws1gt_4),169). orientation(mws1gt_6,146,169). coords(mws1gt_6,73,66,169).  visible(mws1gt_6,169). happensAt(walking(mws1gt_6),169).
orientation(mws1gt_4,0,170). coords(mws1gt_4,29,122,170).  visible(mws1gt_4,170). happensAt(walking(mws1gt_4),170). orientation(mws1gt_6,146,170). coords(mws1gt_6,73,67,170).  visible(mws1gt_6,170). happensAt(walking(mws1gt_6),170).
orientation(mws1gt_4,0,171). coords(mws1gt_4,29,121,171).  visible(mws1gt_4,171). happensAt(walking(mws1gt_4),171). orientation(mws1gt_6,146,171). coords(mws1gt_6,72,67,171).  visible(mws1gt_6,171). happensAt(walking(mws1gt_6),171).
orientation(mws1gt_4,0,172). coords(mws1gt_4,29,121,172).  visible(mws1gt_4,172). happensAt(walking(mws1gt_4),172). orientation(mws1gt_6,146,172). coords(mws1gt_6,72,67,172).  visible(mws1gt_6,172). happensAt(walking(mws1gt_6),172).
orientation(mws1gt_4,0,173). coords(mws1gt_4,29,121,173).  visible(mws1gt_4,173). happensAt(walking(mws1gt_4),173). orientation(mws1gt_6,146,173). coords(mws1gt_6,72,68,173).  visible(mws1gt_6,173). happensAt(walking(mws1gt_6),173).
orientation(mws1gt_4,0,174). coords(mws1gt_4,29,121,174).  visible(mws1gt_4,174). happensAt(walking(mws1gt_4),174). orientation(mws1gt_6,140,174). coords(mws1gt_6,72,68,174).  visible(mws1gt_6,174). happensAt(walking(mws1gt_6),174).
orientation(mws1gt_4,0,175). coords(mws1gt_4,29,120,175).  visible(mws1gt_4,175). happensAt(walking(mws1gt_4),175). orientation(mws1gt_6,140,175). coords(mws1gt_6,71,71,175).  visible(mws1gt_6,175). happensAt(walking(mws1gt_6),175).
orientation(mws1gt_4,0,176). coords(mws1gt_4,29,120,176).  visible(mws1gt_4,176). happensAt(walking(mws1gt_4),176). orientation(mws1gt_6,140,176). coords(mws1gt_6,70,71,176).  visible(mws1gt_6,176). happensAt(walking(mws1gt_6),176).
orientation(mws1gt_4,0,177). coords(mws1gt_4,29,120,177).  visible(mws1gt_4,177). happensAt(walking(mws1gt_4),177). orientation(mws1gt_6,140,177). coords(mws1gt_6,70,72,177).  visible(mws1gt_6,177). happensAt(walking(mws1gt_6),177).
orientation(mws1gt_4,0,178). coords(mws1gt_4,29,119,178).  visible(mws1gt_4,178). happensAt(walking(mws1gt_4),178). orientation(mws1gt_6,140,178). coords(mws1gt_6,70,73,178).  visible(mws1gt_6,178). happensAt(walking(mws1gt_6),178).
orientation(mws1gt_4,0,179). coords(mws1gt_4,29,119,179).  visible(mws1gt_4,179). happensAt(walking(mws1gt_4),179). orientation(mws1gt_6,140,179). coords(mws1gt_6,70,73,179).  visible(mws1gt_6,179). happensAt(walking(mws1gt_6),179).
orientation(mws1gt_4,0,180). coords(mws1gt_4,30,118,180).  visible(mws1gt_4,180). happensAt(walking(mws1gt_4),180). orientation(mws1gt_6,140,180). coords(mws1gt_6,70,73,180).  visible(mws1gt_6,180). happensAt(walking(mws1gt_6),180).
orientation(mws1gt_4,0,181). coords(mws1gt_4,30,118,181).  visible(mws1gt_4,181). happensAt(walking(mws1gt_4),181). orientation(mws1gt_6,140,181). coords(mws1gt_6,69,74,181).  visible(mws1gt_6,181). happensAt(walking(mws1gt_6),181).
orientation(mws1gt_4,0,182). coords(mws1gt_4,30,118,182).  visible(mws1gt_4,182). happensAt(walking(mws1gt_4),182). orientation(mws1gt_6,140,182). coords(mws1gt_6,69,74,182).  visible(mws1gt_6,182). happensAt(walking(mws1gt_6),182).
orientation(mws1gt_4,0,183). coords(mws1gt_4,30,117,183).  visible(mws1gt_4,183). happensAt(walking(mws1gt_4),183). orientation(mws1gt_6,140,183). coords(mws1gt_6,69,75,183).  visible(mws1gt_6,183). happensAt(walking(mws1gt_6),183).
orientation(mws1gt_4,0,184). coords(mws1gt_4,30,117,184).  visible(mws1gt_4,184). happensAt(walking(mws1gt_4),184). orientation(mws1gt_6,140,184). coords(mws1gt_6,68,76,184).  visible(mws1gt_6,184). happensAt(walking(mws1gt_6),184).
orientation(mws1gt_4,0,185). coords(mws1gt_4,30,117,185).  visible(mws1gt_4,185). happensAt(walking(mws1gt_4),185). orientation(mws1gt_6,140,185). coords(mws1gt_6,68,76,185).  visible(mws1gt_6,185). happensAt(walking(mws1gt_6),185).
orientation(mws1gt_4,0,186). coords(mws1gt_4,30,116,186).  visible(mws1gt_4,186). happensAt(walking(mws1gt_4),186). orientation(mws1gt_6,140,186). coords(mws1gt_6,68,76,186).  visible(mws1gt_6,186). happensAt(walking(mws1gt_6),186).
orientation(mws1gt_4,0,187). coords(mws1gt_4,30,115,187).  visible(mws1gt_4,187). happensAt(walking(mws1gt_4),187). orientation(mws1gt_6,140,187). coords(mws1gt_6,68,77,187).  visible(mws1gt_6,187). happensAt(walking(mws1gt_6),187).
orientation(mws1gt_4,0,188). coords(mws1gt_4,30,115,188).  visible(mws1gt_4,188). happensAt(walking(mws1gt_4),188). orientation(mws1gt_6,140,188). coords(mws1gt_6,67,77,188).  visible(mws1gt_6,188). happensAt(walking(mws1gt_6),188).
orientation(mws1gt_4,0,189). coords(mws1gt_4,31,114,189).  visible(mws1gt_4,189). happensAt(walking(mws1gt_4),189). orientation(mws1gt_6,140,189). coords(mws1gt_6,66,78,189).  visible(mws1gt_6,189). happensAt(walking(mws1gt_6),189).
orientation(mws1gt_4,0,190). coords(mws1gt_4,31,114,190).  visible(mws1gt_4,190). happensAt(walking(mws1gt_4),190). orientation(mws1gt_6,140,190). coords(mws1gt_6,66,79,190).  visible(mws1gt_6,190). happensAt(walking(mws1gt_6),190).
orientation(mws1gt_4,0,191). coords(mws1gt_4,31,114,191).  visible(mws1gt_4,191). happensAt(walking(mws1gt_4),191). orientation(mws1gt_6,140,191). coords(mws1gt_6,66,80,191).  visible(mws1gt_6,191). happensAt(walking(mws1gt_6),191).
orientation(mws1gt_4,0,192). coords(mws1gt_4,31,114,192).  visible(mws1gt_4,192). happensAt(walking(mws1gt_4),192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192). happensAt(walking(mws1gt_6),192).
#show holdsAt/2.