
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

	orientation(lb1gt_0,136,168). coords(lb1gt_0,161,53,168).  visible(lb1gt_0,168). happensAt(walking(lb1gt_0),168). orientation(lb1gt_2,144,168). coords(lb1gt_2,111,111,168).  visible(lb1gt_2,168). happensAt(walking(lb1gt_2),168).
orientation(lb1gt_0,136,169). coords(lb1gt_0,161,51,169).  visible(lb1gt_0,169). happensAt(walking(lb1gt_0),169). orientation(lb1gt_2,144,169). coords(lb1gt_2,112,111,169).  visible(lb1gt_2,169). happensAt(walking(lb1gt_2),169).
orientation(lb1gt_0,136,170). coords(lb1gt_0,160,48,170).  visible(lb1gt_0,170). happensAt(walking(lb1gt_0),170). orientation(lb1gt_2,144,170). coords(lb1gt_2,112,111,170).  visible(lb1gt_2,170). happensAt(walking(lb1gt_2),170).
orientation(lb1gt_0,136,171). coords(lb1gt_0,160,47,171).  visible(lb1gt_0,171). happensAt(walking(lb1gt_0),171). orientation(lb1gt_2,144,171). coords(lb1gt_2,112,111,171).  visible(lb1gt_2,171). happensAt(walking(lb1gt_2),171).
orientation(lb1gt_0,136,172). coords(lb1gt_0,161,47,172).  visible(lb1gt_0,172). happensAt(walking(lb1gt_0),172). orientation(lb1gt_2,144,172). coords(lb1gt_2,113,111,172).  visible(lb1gt_2,172). happensAt(walking(lb1gt_2),172).
orientation(lb1gt_0,136,173). coords(lb1gt_0,161,47,173).  visible(lb1gt_0,173). happensAt(walking(lb1gt_0),173). orientation(lb1gt_2,144,173). coords(lb1gt_2,113,111,173).  visible(lb1gt_2,173). happensAt(walking(lb1gt_2),173).
orientation(lb1gt_0,136,174). coords(lb1gt_0,161,46,174).  visible(lb1gt_0,174). happensAt(walking(lb1gt_0),174). orientation(lb1gt_2,144,174). coords(lb1gt_2,113,111,174).  visible(lb1gt_2,174). happensAt(walking(lb1gt_2),174).
orientation(lb1gt_0,136,175). coords(lb1gt_0,161,46,175).  visible(lb1gt_0,175). happensAt(walking(lb1gt_0),175). orientation(lb1gt_2,144,175). coords(lb1gt_2,113,111,175).  visible(lb1gt_2,175). happensAt(walking(lb1gt_2),175).
orientation(lb1gt_0,136,176). coords(lb1gt_0,161,44,176).  visible(lb1gt_0,176). happensAt(walking(lb1gt_0),176). orientation(lb1gt_2,144,176). coords(lb1gt_2,113,111,176).  visible(lb1gt_2,176). happensAt(walking(lb1gt_2),176).
orientation(lb1gt_0,136,177). coords(lb1gt_0,161,43,177).  visible(lb1gt_0,177). happensAt(walking(lb1gt_0),177). orientation(lb1gt_2,144,177). coords(lb1gt_2,114,112,177).  visible(lb1gt_2,177). happensAt(walking(lb1gt_2),177).
orientation(lb1gt_0,136,178). coords(lb1gt_0,161,41,178).  visible(lb1gt_0,178). happensAt(walking(lb1gt_0),178). orientation(lb1gt_2,144,178). coords(lb1gt_2,114,112,178).  visible(lb1gt_2,178). happensAt(walking(lb1gt_2),178).
orientation(lb1gt_0,136,179). coords(lb1gt_0,161,39,179).  visible(lb1gt_0,179). happensAt(walking(lb1gt_0),179). orientation(lb1gt_2,144,179). coords(lb1gt_2,115,112,179).  visible(lb1gt_2,179). happensAt(walking(lb1gt_2),179).
orientation(lb1gt_0,136,180). coords(lb1gt_0,161,38,180).  visible(lb1gt_0,180). happensAt(walking(lb1gt_0),180). orientation(lb1gt_2,144,180). coords(lb1gt_2,115,113,180).  visible(lb1gt_2,180). happensAt(walking(lb1gt_2),180).
orientation(lb1gt_0,136,181). coords(lb1gt_0,162,38,181).  visible(lb1gt_0,181). happensAt(walking(lb1gt_0),181). orientation(lb1gt_2,144,181). coords(lb1gt_2,116,113,181).  visible(lb1gt_2,181). happensAt(walking(lb1gt_2),181).
orientation(lb1gt_0,136,182). coords(lb1gt_0,162,38,182).  visible(lb1gt_0,182). happensAt(walking(lb1gt_0),182). orientation(lb1gt_2,144,182). coords(lb1gt_2,117,113,182).  visible(lb1gt_2,182). happensAt(walking(lb1gt_2),182).
orientation(lb1gt_0,136,183). coords(lb1gt_0,162,38,183).  visible(lb1gt_0,183). happensAt(walking(lb1gt_0),183). orientation(lb1gt_2,144,183). coords(lb1gt_2,117,114,183).  visible(lb1gt_2,183). happensAt(walking(lb1gt_2),183).
orientation(lb1gt_0,136,184). coords(lb1gt_0,163,38,184).  visible(lb1gt_0,184). happensAt(walking(lb1gt_0),184). orientation(lb1gt_2,144,184). coords(lb1gt_2,117,114,184).  visible(lb1gt_2,184). happensAt(walking(lb1gt_2),184).
orientation(lb1gt_0,136,185). coords(lb1gt_0,163,37,185).  visible(lb1gt_0,185). happensAt(walking(lb1gt_0),185). orientation(lb1gt_2,144,185). coords(lb1gt_2,119,114,185).  visible(lb1gt_2,185). happensAt(walking(lb1gt_2),185).
orientation(lb1gt_0,136,186). coords(lb1gt_0,164,37,186).  visible(lb1gt_0,186). happensAt(walking(lb1gt_0),186). orientation(lb1gt_2,144,186). coords(lb1gt_2,119,114,186).  visible(lb1gt_2,186). happensAt(walking(lb1gt_2),186).
orientation(lb1gt_0,136,187). coords(lb1gt_0,165,35,187).  visible(lb1gt_0,187). happensAt(walking(lb1gt_0),187). orientation(lb1gt_2,144,187). coords(lb1gt_2,119,115,187).  visible(lb1gt_2,187). happensAt(walking(lb1gt_2),187).
orientation(lb1gt_0,136,188). coords(lb1gt_0,165,35,188).  visible(lb1gt_0,188). happensAt(walking(lb1gt_0),188). orientation(lb1gt_2,144,188). coords(lb1gt_2,119,115,188).  visible(lb1gt_2,188). happensAt(walking(lb1gt_2),188).
orientation(lb1gt_0,136,189). coords(lb1gt_0,165,34,189).  visible(lb1gt_0,189). happensAt(walking(lb1gt_0),189). orientation(lb1gt_2,144,189). coords(lb1gt_2,119,115,189).  visible(lb1gt_2,189). happensAt(walking(lb1gt_2),189).
orientation(lb1gt_0,136,190). coords(lb1gt_0,166,34,190).  visible(lb1gt_0,190). happensAt(walking(lb1gt_0),190). orientation(lb1gt_2,144,190). coords(lb1gt_2,119,115,190).  visible(lb1gt_2,190). happensAt(walking(lb1gt_2),190).
orientation(lb1gt_0,136,191). coords(lb1gt_0,166,34,191).  visible(lb1gt_0,191). happensAt(walking(lb1gt_0),191). orientation(lb1gt_2,144,191). coords(lb1gt_2,120,115,191).  visible(lb1gt_2,191). happensAt(walking(lb1gt_2),191).
orientation(lb1gt_0,136,192). coords(lb1gt_0,166,33,192).  visible(lb1gt_0,192). happensAt(walking(lb1gt_0),192). orientation(lb1gt_2,144,192). coords(lb1gt_2,120,115,192).  visible(lb1gt_2,192). happensAt(walking(lb1gt_2),192).
#show holdsAt/2.