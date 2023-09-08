
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

	orientation(wk2gt_1,129,166). coords(wk2gt_1,115,53,166).  visible(wk2gt_1,166). happensAt(inactive(wk2gt_1),166). orientation(wk2gt_3,145,166). coords(wk2gt_3,196,166,166).  visible(wk2gt_3,166). happensAt(active(wk2gt_3),166).
orientation(wk2gt_1,129,167). coords(wk2gt_1,115,53,167).  visible(wk2gt_1,167). happensAt(inactive(wk2gt_1),167). orientation(wk2gt_3,145,167). coords(wk2gt_3,198,167,167).  visible(wk2gt_3,167). happensAt(active(wk2gt_3),167).
orientation(wk2gt_1,129,168). coords(wk2gt_1,115,53,168).  visible(wk2gt_1,168). happensAt(inactive(wk2gt_1),168). orientation(wk2gt_3,145,168). coords(wk2gt_3,200,168,168).  visible(wk2gt_3,168). happensAt(active(wk2gt_3),168).
orientation(wk2gt_1,129,169). coords(wk2gt_1,115,53,169).  visible(wk2gt_1,169). happensAt(inactive(wk2gt_1),169). orientation(wk2gt_3,145,169). coords(wk2gt_3,201,169,169).  visible(wk2gt_3,169). happensAt(active(wk2gt_3),169).
orientation(wk2gt_1,129,170). coords(wk2gt_1,115,53,170).  visible(wk2gt_1,170). happensAt(inactive(wk2gt_1),170). orientation(wk2gt_3,145,170). coords(wk2gt_3,202,170,170).  visible(wk2gt_3,170). happensAt(active(wk2gt_3),170).
orientation(wk2gt_1,129,171). coords(wk2gt_1,115,53,171).  visible(wk2gt_1,171). happensAt(inactive(wk2gt_1),171). orientation(wk2gt_3,145,171). coords(wk2gt_3,203,172,171).  visible(wk2gt_3,171). happensAt(active(wk2gt_3),171).
orientation(wk2gt_1,129,172). coords(wk2gt_1,115,53,172).  visible(wk2gt_1,172). happensAt(inactive(wk2gt_1),172). orientation(wk2gt_3,145,172). coords(wk2gt_3,205,173,172).  visible(wk2gt_3,172). happensAt(active(wk2gt_3),172).
orientation(wk2gt_1,129,173). coords(wk2gt_1,115,53,173).  visible(wk2gt_1,173). happensAt(inactive(wk2gt_1),173). orientation(wk2gt_3,161,173). coords(wk2gt_3,206,175,173).  visible(wk2gt_3,173). happensAt(active(wk2gt_3),173).
orientation(wk2gt_1,129,174). coords(wk2gt_1,115,53,174).  visible(wk2gt_1,174). happensAt(inactive(wk2gt_1),174). orientation(wk2gt_3,164,174). coords(wk2gt_3,206,175,174).  visible(wk2gt_3,174). happensAt(active(wk2gt_3),174).
orientation(wk2gt_1,129,175). coords(wk2gt_1,115,53,175).  visible(wk2gt_1,175). happensAt(inactive(wk2gt_1),175). orientation(wk2gt_3,164,175). coords(wk2gt_3,207,176,175).  visible(wk2gt_3,175). happensAt(walking(wk2gt_3),175).
orientation(wk2gt_1,129,176). coords(wk2gt_1,115,53,176).  visible(wk2gt_1,176). happensAt(inactive(wk2gt_1),176). orientation(wk2gt_3,164,176). coords(wk2gt_3,208,179,176).  visible(wk2gt_3,176). happensAt(walking(wk2gt_3),176).
orientation(wk2gt_1,129,177). coords(wk2gt_1,115,53,177).  visible(wk2gt_1,177). happensAt(inactive(wk2gt_1),177). orientation(wk2gt_3,164,177). coords(wk2gt_3,210,181,177).  visible(wk2gt_3,177). happensAt(walking(wk2gt_3),177).
orientation(wk2gt_1,129,178). coords(wk2gt_1,115,53,178).  visible(wk2gt_1,178). happensAt(inactive(wk2gt_1),178). orientation(wk2gt_3,164,178). coords(wk2gt_3,210,183,178).  visible(wk2gt_3,178). happensAt(walking(wk2gt_3),178).
orientation(wk2gt_1,129,179). coords(wk2gt_1,115,53,179).  visible(wk2gt_1,179). happensAt(inactive(wk2gt_1),179). orientation(wk2gt_3,164,179). coords(wk2gt_3,213,185,179).  visible(wk2gt_3,179). happensAt(walking(wk2gt_3),179).
orientation(wk2gt_1,129,180). coords(wk2gt_1,115,53,180).  visible(wk2gt_1,180). happensAt(inactive(wk2gt_1),180). orientation(wk2gt_3,169,180). coords(wk2gt_3,217,187,180).  visible(wk2gt_3,180). happensAt(walking(wk2gt_3),180).
orientation(wk2gt_1,129,181). coords(wk2gt_1,115,53,181).  visible(wk2gt_1,181). happensAt(inactive(wk2gt_1),181). orientation(wk2gt_3,169,181). coords(wk2gt_3,220,189,181).  visible(wk2gt_3,181). happensAt(walking(wk2gt_3),181).
orientation(wk2gt_1,129,182). coords(wk2gt_1,115,53,182).  visible(wk2gt_1,182). happensAt(inactive(wk2gt_1),182). orientation(wk2gt_3,169,182). coords(wk2gt_3,223,191,182).  visible(wk2gt_3,182). happensAt(walking(wk2gt_3),182).
orientation(wk2gt_1,129,183). coords(wk2gt_1,115,53,183).  visible(wk2gt_1,183). happensAt(inactive(wk2gt_1),183). orientation(wk2gt_3,169,183). coords(wk2gt_3,225,194,183).  visible(wk2gt_3,183). happensAt(walking(wk2gt_3),183).
orientation(wk2gt_1,129,184). coords(wk2gt_1,115,53,184).  visible(wk2gt_1,184). happensAt(inactive(wk2gt_1),184). orientation(wk2gt_3,169,184). coords(wk2gt_3,227,197,184).  visible(wk2gt_3,184). happensAt(walking(wk2gt_3),184).
orientation(wk2gt_1,129,185). coords(wk2gt_1,115,53,185).  visible(wk2gt_1,185). happensAt(inactive(wk2gt_1),185). orientation(wk2gt_3,165,185). coords(wk2gt_3,230,200,185).  visible(wk2gt_3,185). happensAt(walking(wk2gt_3),185).
orientation(wk2gt_1,129,186). coords(wk2gt_1,115,53,186).  visible(wk2gt_1,186). happensAt(inactive(wk2gt_1),186). orientation(wk2gt_3,165,186). coords(wk2gt_3,232,202,186).  visible(wk2gt_3,186). happensAt(walking(wk2gt_3),186).
orientation(wk2gt_1,129,187). coords(wk2gt_1,115,53,187).  visible(wk2gt_1,187). happensAt(inactive(wk2gt_1),187). orientation(wk2gt_3,165,187). coords(wk2gt_3,233,205,187).  visible(wk2gt_3,187). happensAt(walking(wk2gt_3),187).
orientation(wk2gt_1,129,188). coords(wk2gt_1,115,53,188).  visible(wk2gt_1,188). happensAt(inactive(wk2gt_1),188). orientation(wk2gt_3,157,188). coords(wk2gt_3,234,208,188).  visible(wk2gt_3,188). happensAt(walking(wk2gt_3),188).
orientation(wk2gt_1,129,189). coords(wk2gt_1,115,53,189).  visible(wk2gt_1,189). happensAt(inactive(wk2gt_1),189). orientation(wk2gt_3,157,189). coords(wk2gt_3,235,211,189).  visible(wk2gt_3,189). happensAt(walking(wk2gt_3),189).
orientation(wk2gt_1,129,190). coords(wk2gt_1,115,53,190).  visible(wk2gt_1,190). happensAt(inactive(wk2gt_1),190). orientation(wk2gt_3,157,190). coords(wk2gt_3,237,216,190).  visible(wk2gt_3,190). happensAt(walking(wk2gt_3),190).
#show holdsAt/2.