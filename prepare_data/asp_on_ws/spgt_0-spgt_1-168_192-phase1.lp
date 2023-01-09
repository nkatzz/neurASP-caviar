
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

	orientation(spgt_0,148,168). coords(spgt_0,111,116,168).  visible(spgt_0,168). happensAt(walking(spgt_0),168). orientation(spgt_1,148,168). coords(spgt_1,108,104,168).  visible(spgt_1,168). happensAt(active(spgt_1),168).
orientation(spgt_0,148,169). coords(spgt_0,112,116,169).  visible(spgt_0,169). happensAt(walking(spgt_0),169). orientation(spgt_1,148,169). coords(spgt_1,108,103,169).  visible(spgt_1,169). happensAt(active(spgt_1),169).
orientation(spgt_0,148,170). coords(spgt_0,112,116,170).  visible(spgt_0,170). happensAt(active(spgt_0),170). orientation(spgt_1,148,170). coords(spgt_1,108,103,170).  visible(spgt_1,170). happensAt(active(spgt_1),170).
orientation(spgt_0,148,171). coords(spgt_0,112,116,171).  visible(spgt_0,171). happensAt(active(spgt_0),171). orientation(spgt_1,148,171). coords(spgt_1,108,103,171).  visible(spgt_1,171). happensAt(active(spgt_1),171).
orientation(spgt_0,148,172). coords(spgt_0,112,116,172).  visible(spgt_0,172). happensAt(active(spgt_0),172). orientation(spgt_1,148,172). coords(spgt_1,108,103,172).  visible(spgt_1,172). happensAt(active(spgt_1),172).
orientation(spgt_0,148,173). coords(spgt_0,112,116,173).  visible(spgt_0,173). happensAt(active(spgt_0),173). orientation(spgt_1,148,173). coords(spgt_1,108,103,173).  visible(spgt_1,173). happensAt(active(spgt_1),173).
orientation(spgt_0,148,174). coords(spgt_0,113,115,174).  visible(spgt_0,174). happensAt(active(spgt_0),174). orientation(spgt_1,148,174). coords(spgt_1,108,103,174).  visible(spgt_1,174). happensAt(active(spgt_1),174).
orientation(spgt_0,148,175). coords(spgt_0,113,115,175).  visible(spgt_0,175). happensAt(active(spgt_0),175). orientation(spgt_1,148,175). coords(spgt_1,108,103,175).  visible(spgt_1,175). happensAt(active(spgt_1),175).
orientation(spgt_0,148,176). coords(spgt_0,113,114,176).  visible(spgt_0,176). happensAt(active(spgt_0),176). orientation(spgt_1,148,176). coords(spgt_1,108,103,176).  visible(spgt_1,176). happensAt(active(spgt_1),176).
orientation(spgt_0,148,177). coords(spgt_0,113,114,177).  visible(spgt_0,177). happensAt(active(spgt_0),177). orientation(spgt_1,148,177). coords(spgt_1,108,103,177).  visible(spgt_1,177). happensAt(active(spgt_1),177).
orientation(spgt_0,148,178). coords(spgt_0,113,114,178).  visible(spgt_0,178). happensAt(active(spgt_0),178). orientation(spgt_1,148,178). coords(spgt_1,108,103,178).  visible(spgt_1,178). happensAt(active(spgt_1),178).
orientation(spgt_0,148,179). coords(spgt_0,113,114,179).  visible(spgt_0,179). happensAt(active(spgt_0),179). orientation(spgt_1,148,179). coords(spgt_1,108,103,179).  visible(spgt_1,179). happensAt(active(spgt_1),179).
orientation(spgt_0,148,180). coords(spgt_0,113,114,180).  visible(spgt_0,180). happensAt(active(spgt_0),180). orientation(spgt_1,148,180). coords(spgt_1,108,103,180).  visible(spgt_1,180). happensAt(active(spgt_1),180).
orientation(spgt_0,148,181). coords(spgt_0,113,114,181).  visible(spgt_0,181). happensAt(inactive(spgt_0),181). orientation(spgt_1,148,181). coords(spgt_1,108,103,181).  visible(spgt_1,181). happensAt(active(spgt_1),181).
orientation(spgt_0,148,182). coords(spgt_0,113,114,182).  visible(spgt_0,182). happensAt(inactive(spgt_0),182). orientation(spgt_1,148,182). coords(spgt_1,108,103,182).  visible(spgt_1,182). happensAt(active(spgt_1),182).
orientation(spgt_0,148,183). coords(spgt_0,113,114,183).  visible(spgt_0,183). happensAt(inactive(spgt_0),183). orientation(spgt_1,148,183). coords(spgt_1,108,103,183).  visible(spgt_1,183). happensAt(active(spgt_1),183).
orientation(spgt_0,148,184). coords(spgt_0,113,114,184).  visible(spgt_0,184). happensAt(inactive(spgt_0),184). orientation(spgt_1,148,184). coords(spgt_1,108,103,184).  visible(spgt_1,184). happensAt(active(spgt_1),184).
orientation(spgt_0,148,185). coords(spgt_0,113,114,185).  visible(spgt_0,185). happensAt(inactive(spgt_0),185). orientation(spgt_1,148,185). coords(spgt_1,108,103,185).  visible(spgt_1,185). happensAt(active(spgt_1),185).
orientation(spgt_0,148,186). coords(spgt_0,113,114,186).  visible(spgt_0,186). happensAt(inactive(spgt_0),186). orientation(spgt_1,148,186). coords(spgt_1,108,103,186).  visible(spgt_1,186). happensAt(active(spgt_1),186).
orientation(spgt_0,148,187). coords(spgt_0,113,114,187).  visible(spgt_0,187). happensAt(inactive(spgt_0),187). orientation(spgt_1,148,187). coords(spgt_1,108,103,187).  visible(spgt_1,187). happensAt(active(spgt_1),187).
orientation(spgt_0,148,188). coords(spgt_0,113,114,188).  visible(spgt_0,188). happensAt(inactive(spgt_0),188). orientation(spgt_1,148,188). coords(spgt_1,108,103,188).  visible(spgt_1,188). happensAt(active(spgt_1),188).
orientation(spgt_0,148,189). coords(spgt_0,113,114,189).  visible(spgt_0,189). happensAt(inactive(spgt_0),189). orientation(spgt_1,148,189). coords(spgt_1,108,103,189).  visible(spgt_1,189). happensAt(active(spgt_1),189).
orientation(spgt_0,148,190). coords(spgt_0,113,114,190).  visible(spgt_0,190). happensAt(inactive(spgt_0),190). orientation(spgt_1,148,190). coords(spgt_1,108,103,190).  visible(spgt_1,190). happensAt(active(spgt_1),190).
orientation(spgt_0,148,191). coords(spgt_0,113,114,191).  visible(spgt_0,191). happensAt(inactive(spgt_0),191). orientation(spgt_1,148,191). coords(spgt_1,108,103,191).  visible(spgt_1,191). happensAt(active(spgt_1),191).
orientation(spgt_0,148,192). coords(spgt_0,113,114,192).  visible(spgt_0,192). happensAt(inactive(spgt_0),192). orientation(spgt_1,148,192). coords(spgt_1,108,103,192).  visible(spgt_1,192). happensAt(active(spgt_1),192).
#show holdsAt/2.