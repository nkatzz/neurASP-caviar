
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

	orientation(wk2gt_1,129,168). coords(wk2gt_1,115,53,168).  visible(wk2gt_1,168). happensAt(inactive(wk2gt_1),168). orientation(wk2gt_2,141,168). coords(wk2gt_2,98,62,168).  visible(wk2gt_2,168). happensAt(inactive(wk2gt_2),168).
orientation(wk2gt_1,129,169). coords(wk2gt_1,115,53,169).  visible(wk2gt_1,169). happensAt(inactive(wk2gt_1),169). orientation(wk2gt_2,141,169). coords(wk2gt_2,98,62,169).  visible(wk2gt_2,169). happensAt(inactive(wk2gt_2),169).
orientation(wk2gt_1,129,170). coords(wk2gt_1,115,53,170).  visible(wk2gt_1,170). happensAt(inactive(wk2gt_1),170). orientation(wk2gt_2,141,170). coords(wk2gt_2,98,62,170).  visible(wk2gt_2,170). happensAt(inactive(wk2gt_2),170).
orientation(wk2gt_1,129,171). coords(wk2gt_1,115,53,171).  visible(wk2gt_1,171). happensAt(inactive(wk2gt_1),171). orientation(wk2gt_2,141,171). coords(wk2gt_2,98,62,171).  visible(wk2gt_2,171). happensAt(inactive(wk2gt_2),171).
orientation(wk2gt_1,129,172). coords(wk2gt_1,115,53,172).  visible(wk2gt_1,172). happensAt(inactive(wk2gt_1),172). orientation(wk2gt_2,141,172). coords(wk2gt_2,98,62,172).  visible(wk2gt_2,172). happensAt(inactive(wk2gt_2),172).
orientation(wk2gt_1,129,173). coords(wk2gt_1,115,53,173).  visible(wk2gt_1,173). happensAt(inactive(wk2gt_1),173). orientation(wk2gt_2,141,173). coords(wk2gt_2,98,62,173).  visible(wk2gt_2,173). happensAt(inactive(wk2gt_2),173).
orientation(wk2gt_1,129,174). coords(wk2gt_1,115,53,174).  visible(wk2gt_1,174). happensAt(inactive(wk2gt_1),174). orientation(wk2gt_2,141,174). coords(wk2gt_2,98,62,174).  visible(wk2gt_2,174). happensAt(inactive(wk2gt_2),174).
orientation(wk2gt_1,129,175). coords(wk2gt_1,115,53,175).  visible(wk2gt_1,175). happensAt(inactive(wk2gt_1),175). orientation(wk2gt_2,141,175). coords(wk2gt_2,98,62,175).  visible(wk2gt_2,175). happensAt(inactive(wk2gt_2),175).
orientation(wk2gt_1,129,176). coords(wk2gt_1,115,53,176).  visible(wk2gt_1,176). happensAt(inactive(wk2gt_1),176). orientation(wk2gt_2,141,176). coords(wk2gt_2,98,62,176).  visible(wk2gt_2,176). happensAt(inactive(wk2gt_2),176).
orientation(wk2gt_1,129,177). coords(wk2gt_1,115,53,177).  visible(wk2gt_1,177). happensAt(inactive(wk2gt_1),177). orientation(wk2gt_2,141,177). coords(wk2gt_2,98,62,177).  visible(wk2gt_2,177). happensAt(inactive(wk2gt_2),177).
orientation(wk2gt_1,129,178). coords(wk2gt_1,115,53,178).  visible(wk2gt_1,178). happensAt(inactive(wk2gt_1),178). orientation(wk2gt_2,141,178). coords(wk2gt_2,98,62,178).  visible(wk2gt_2,178). happensAt(inactive(wk2gt_2),178).
orientation(wk2gt_1,129,179). coords(wk2gt_1,115,53,179).  visible(wk2gt_1,179). happensAt(inactive(wk2gt_1),179). orientation(wk2gt_2,141,179). coords(wk2gt_2,98,62,179).  visible(wk2gt_2,179). happensAt(inactive(wk2gt_2),179).
orientation(wk2gt_1,129,180). coords(wk2gt_1,115,53,180).  visible(wk2gt_1,180). happensAt(inactive(wk2gt_1),180). orientation(wk2gt_2,141,180). coords(wk2gt_2,98,62,180).  visible(wk2gt_2,180). happensAt(inactive(wk2gt_2),180).
orientation(wk2gt_1,129,181). coords(wk2gt_1,115,53,181).  visible(wk2gt_1,181). happensAt(inactive(wk2gt_1),181). orientation(wk2gt_2,141,181). coords(wk2gt_2,98,62,181).  visible(wk2gt_2,181). happensAt(inactive(wk2gt_2),181).
orientation(wk2gt_1,129,182). coords(wk2gt_1,115,53,182).  visible(wk2gt_1,182). happensAt(inactive(wk2gt_1),182). orientation(wk2gt_2,141,182). coords(wk2gt_2,98,62,182).  visible(wk2gt_2,182). happensAt(inactive(wk2gt_2),182).
orientation(wk2gt_1,129,183). coords(wk2gt_1,115,53,183).  visible(wk2gt_1,183). happensAt(inactive(wk2gt_1),183). orientation(wk2gt_2,141,183). coords(wk2gt_2,98,62,183).  visible(wk2gt_2,183). happensAt(inactive(wk2gt_2),183).
orientation(wk2gt_1,129,184). coords(wk2gt_1,115,53,184).  visible(wk2gt_1,184). happensAt(inactive(wk2gt_1),184). orientation(wk2gt_2,141,184). coords(wk2gt_2,98,62,184).  visible(wk2gt_2,184). happensAt(inactive(wk2gt_2),184).
orientation(wk2gt_1,129,185). coords(wk2gt_1,115,53,185).  visible(wk2gt_1,185). happensAt(inactive(wk2gt_1),185). orientation(wk2gt_2,141,185). coords(wk2gt_2,98,62,185).  visible(wk2gt_2,185). happensAt(inactive(wk2gt_2),185).
orientation(wk2gt_1,129,186). coords(wk2gt_1,115,53,186).  visible(wk2gt_1,186). happensAt(inactive(wk2gt_1),186). orientation(wk2gt_2,141,186). coords(wk2gt_2,98,62,186).  visible(wk2gt_2,186). happensAt(inactive(wk2gt_2),186).
orientation(wk2gt_1,129,187). coords(wk2gt_1,115,53,187).  visible(wk2gt_1,187). happensAt(inactive(wk2gt_1),187). orientation(wk2gt_2,141,187). coords(wk2gt_2,98,62,187).  visible(wk2gt_2,187). happensAt(inactive(wk2gt_2),187).
orientation(wk2gt_1,129,188). coords(wk2gt_1,115,53,188).  visible(wk2gt_1,188). happensAt(inactive(wk2gt_1),188). orientation(wk2gt_2,141,188). coords(wk2gt_2,98,62,188).  visible(wk2gt_2,188). happensAt(inactive(wk2gt_2),188).
orientation(wk2gt_1,129,189). coords(wk2gt_1,115,53,189).  visible(wk2gt_1,189). happensAt(inactive(wk2gt_1),189). orientation(wk2gt_2,141,189). coords(wk2gt_2,98,62,189).  visible(wk2gt_2,189). happensAt(inactive(wk2gt_2),189).
orientation(wk2gt_1,129,190). coords(wk2gt_1,115,53,190).  visible(wk2gt_1,190). happensAt(inactive(wk2gt_1),190). orientation(wk2gt_2,141,190). coords(wk2gt_2,98,62,190).  visible(wk2gt_2,190). happensAt(inactive(wk2gt_2),190).
orientation(wk2gt_1,129,191). coords(wk2gt_1,115,53,191).  visible(wk2gt_1,191). happensAt(inactive(wk2gt_1),191). orientation(wk2gt_2,141,191). coords(wk2gt_2,98,62,191).  visible(wk2gt_2,191). happensAt(inactive(wk2gt_2),191).
orientation(wk2gt_1,129,192). coords(wk2gt_1,115,53,192).  visible(wk2gt_1,192). happensAt(inactive(wk2gt_1),192). orientation(wk2gt_2,141,192). coords(wk2gt_2,98,62,192).  visible(wk2gt_2,192). happensAt(inactive(wk2gt_2),192).
#show holdsAt/2.