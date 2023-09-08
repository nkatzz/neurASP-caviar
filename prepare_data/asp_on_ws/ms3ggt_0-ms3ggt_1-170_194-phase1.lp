
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

	orientation(ms3ggt_0,148,170). coords(ms3ggt_0,68,130,170).  visible(ms3ggt_0,170). happensAt(walking(ms3ggt_0),170). orientation(ms3ggt_1,162,170). coords(ms3ggt_1,32,161,170).  happensAt(appear(ms3ggt_1),170). happensAt(walking(ms3ggt_1),170).
orientation(ms3ggt_0,148,171). coords(ms3ggt_0,68,130,171).  visible(ms3ggt_0,171). happensAt(walking(ms3ggt_0),171). orientation(ms3ggt_1,162,171). coords(ms3ggt_1,32,161,171).  visible(ms3ggt_1,171). happensAt(walking(ms3ggt_1),171).
orientation(ms3ggt_0,148,172). coords(ms3ggt_0,67,131,172).  visible(ms3ggt_0,172). happensAt(walking(ms3ggt_0),172). orientation(ms3ggt_1,162,172). coords(ms3ggt_1,32,161,172).  visible(ms3ggt_1,172). happensAt(walking(ms3ggt_1),172).
orientation(ms3ggt_0,148,173). coords(ms3ggt_0,67,131,173).  visible(ms3ggt_0,173). happensAt(walking(ms3ggt_0),173). orientation(ms3ggt_1,162,173). coords(ms3ggt_1,32,161,173).  visible(ms3ggt_1,173). happensAt(walking(ms3ggt_1),173).
orientation(ms3ggt_0,148,174). coords(ms3ggt_0,67,132,174).  visible(ms3ggt_0,174). happensAt(walking(ms3ggt_0),174). orientation(ms3ggt_1,162,174). coords(ms3ggt_1,33,161,174).  visible(ms3ggt_1,174). happensAt(walking(ms3ggt_1),174).
orientation(ms3ggt_0,148,175). coords(ms3ggt_0,66,132,175).  visible(ms3ggt_0,175). happensAt(walking(ms3ggt_0),175). orientation(ms3ggt_1,162,175). coords(ms3ggt_1,34,161,175).  visible(ms3ggt_1,175). happensAt(walking(ms3ggt_1),175).
orientation(ms3ggt_0,159,176). coords(ms3ggt_0,66,132,176).  visible(ms3ggt_0,176). happensAt(walking(ms3ggt_0),176). orientation(ms3ggt_1,162,176). coords(ms3ggt_1,34,161,176).  visible(ms3ggt_1,176). happensAt(walking(ms3ggt_1),176).
orientation(ms3ggt_0,149,177). coords(ms3ggt_0,66,132,177).  visible(ms3ggt_0,177). happensAt(walking(ms3ggt_0),177). orientation(ms3ggt_1,162,177). coords(ms3ggt_1,34,161,177).  visible(ms3ggt_1,177). happensAt(walking(ms3ggt_1),177).
orientation(ms3ggt_0,149,178). coords(ms3ggt_0,66,132,178).  visible(ms3ggt_0,178). happensAt(walking(ms3ggt_0),178). orientation(ms3ggt_1,162,178). coords(ms3ggt_1,34,161,178).  visible(ms3ggt_1,178). happensAt(walking(ms3ggt_1),178).
orientation(ms3ggt_0,149,179). coords(ms3ggt_0,66,133,179).  visible(ms3ggt_0,179). happensAt(walking(ms3ggt_0),179). orientation(ms3ggt_1,162,179). coords(ms3ggt_1,34,161,179).  visible(ms3ggt_1,179). happensAt(walking(ms3ggt_1),179).
orientation(ms3ggt_0,149,180). coords(ms3ggt_0,66,134,180).  visible(ms3ggt_0,180). happensAt(walking(ms3ggt_0),180). orientation(ms3ggt_1,162,180). coords(ms3ggt_1,34,161,180).  visible(ms3ggt_1,180). happensAt(walking(ms3ggt_1),180).
orientation(ms3ggt_0,149,181). coords(ms3ggt_0,66,135,181).  visible(ms3ggt_0,181). happensAt(walking(ms3ggt_0),181). orientation(ms3ggt_1,162,181). coords(ms3ggt_1,34,162,181).  visible(ms3ggt_1,181). happensAt(walking(ms3ggt_1),181).
orientation(ms3ggt_0,149,182). coords(ms3ggt_0,66,135,182).  visible(ms3ggt_0,182). happensAt(walking(ms3ggt_0),182). orientation(ms3ggt_1,162,182). coords(ms3ggt_1,35,162,182).  visible(ms3ggt_1,182). happensAt(walking(ms3ggt_1),182).
orientation(ms3ggt_0,149,183). coords(ms3ggt_0,66,135,183).  visible(ms3ggt_0,183). happensAt(walking(ms3ggt_0),183). orientation(ms3ggt_1,162,183). coords(ms3ggt_1,35,162,183).  visible(ms3ggt_1,183). happensAt(walking(ms3ggt_1),183).
orientation(ms3ggt_0,149,184). coords(ms3ggt_0,65,136,184).  visible(ms3ggt_0,184). happensAt(walking(ms3ggt_0),184). orientation(ms3ggt_1,162,184). coords(ms3ggt_1,35,162,184).  visible(ms3ggt_1,184). happensAt(walking(ms3ggt_1),184).
orientation(ms3ggt_0,149,185). coords(ms3ggt_0,65,138,185).  visible(ms3ggt_0,185). happensAt(walking(ms3ggt_0),185). orientation(ms3ggt_1,162,185). coords(ms3ggt_1,35,162,185).  visible(ms3ggt_1,185). happensAt(walking(ms3ggt_1),185).
orientation(ms3ggt_0,149,186). coords(ms3ggt_0,65,138,186).  visible(ms3ggt_0,186). happensAt(walking(ms3ggt_0),186). orientation(ms3ggt_1,162,186). coords(ms3ggt_1,35,162,186).  visible(ms3ggt_1,186). happensAt(walking(ms3ggt_1),186).
orientation(ms3ggt_0,149,187). coords(ms3ggt_0,65,140,187).  visible(ms3ggt_0,187). happensAt(walking(ms3ggt_0),187). orientation(ms3ggt_1,162,187). coords(ms3ggt_1,35,162,187).  visible(ms3ggt_1,187). happensAt(walking(ms3ggt_1),187).
orientation(ms3ggt_0,149,188). coords(ms3ggt_0,65,140,188).  visible(ms3ggt_0,188). happensAt(walking(ms3ggt_0),188). orientation(ms3ggt_1,164,188). coords(ms3ggt_1,36,162,188).  visible(ms3ggt_1,188). happensAt(walking(ms3ggt_1),188).
orientation(ms3ggt_0,149,189). coords(ms3ggt_0,65,141,189).  visible(ms3ggt_0,189). happensAt(walking(ms3ggt_0),189). orientation(ms3ggt_1,164,189). coords(ms3ggt_1,36,162,189).  visible(ms3ggt_1,189). happensAt(walking(ms3ggt_1),189).
orientation(ms3ggt_0,149,190). coords(ms3ggt_0,64,141,190).  visible(ms3ggt_0,190). happensAt(walking(ms3ggt_0),190). orientation(ms3ggt_1,164,190). coords(ms3ggt_1,36,162,190).  visible(ms3ggt_1,190). happensAt(walking(ms3ggt_1),190).
orientation(ms3ggt_0,149,191). coords(ms3ggt_0,64,142,191).  visible(ms3ggt_0,191). happensAt(walking(ms3ggt_0),191). orientation(ms3ggt_1,164,191). coords(ms3ggt_1,36,162,191).  visible(ms3ggt_1,191). happensAt(walking(ms3ggt_1),191).
orientation(ms3ggt_0,149,192). coords(ms3ggt_0,64,142,192).  visible(ms3ggt_0,192). happensAt(walking(ms3ggt_0),192). orientation(ms3ggt_1,154,192). coords(ms3ggt_1,37,164,192).  visible(ms3ggt_1,192). happensAt(walking(ms3ggt_1),192).
orientation(ms3ggt_0,149,193). coords(ms3ggt_0,64,142,193).  visible(ms3ggt_0,193). happensAt(walking(ms3ggt_0),193). orientation(ms3ggt_1,154,193). coords(ms3ggt_1,37,164,193).  visible(ms3ggt_1,193). happensAt(walking(ms3ggt_1),193).
orientation(ms3ggt_0,157,194). coords(ms3ggt_0,63,142,194).  visible(ms3ggt_0,194). happensAt(walking(ms3ggt_0),194). orientation(ms3ggt_1,154,194). coords(ms3ggt_1,37,164,194).  visible(ms3ggt_1,194). happensAt(walking(ms3ggt_1),194).
#show holdsAt/2.