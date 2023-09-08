
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

	orientation(lb1gt_0,136,168). coords(lb1gt_0,161,53,168).  visible(lb1gt_0,168). happensAt(walking(lb1gt_0),168). orientation(lb1gt_1,151,168). coords(lb1gt_1,101,147,168).  visible(lb1gt_1,168). happensAt(walking(lb1gt_1),168).
orientation(lb1gt_0,136,169). coords(lb1gt_0,161,51,169).  visible(lb1gt_0,169). happensAt(walking(lb1gt_0),169). orientation(lb1gt_1,151,169). coords(lb1gt_1,101,147,169).  visible(lb1gt_1,169). happensAt(walking(lb1gt_1),169).
orientation(lb1gt_0,136,170). coords(lb1gt_0,160,48,170).  visible(lb1gt_0,170). happensAt(walking(lb1gt_0),170). orientation(lb1gt_1,151,170). coords(lb1gt_1,101,148,170).  visible(lb1gt_1,170). happensAt(walking(lb1gt_1),170).
orientation(lb1gt_0,136,171). coords(lb1gt_0,160,47,171).  visible(lb1gt_0,171). happensAt(walking(lb1gt_0),171). orientation(lb1gt_1,151,171). coords(lb1gt_1,101,148,171).  visible(lb1gt_1,171). happensAt(walking(lb1gt_1),171).
orientation(lb1gt_0,136,172). coords(lb1gt_0,161,47,172).  visible(lb1gt_0,172). happensAt(walking(lb1gt_0),172). orientation(lb1gt_1,151,172). coords(lb1gt_1,102,148,172).  visible(lb1gt_1,172). happensAt(walking(lb1gt_1),172).
orientation(lb1gt_0,136,173). coords(lb1gt_0,161,47,173).  visible(lb1gt_0,173). happensAt(walking(lb1gt_0),173). orientation(lb1gt_1,151,173). coords(lb1gt_1,102,148,173).  visible(lb1gt_1,173). happensAt(walking(lb1gt_1),173).
orientation(lb1gt_0,136,174). coords(lb1gt_0,161,46,174).  visible(lb1gt_0,174). happensAt(walking(lb1gt_0),174). orientation(lb1gt_1,151,174). coords(lb1gt_1,102,148,174).  visible(lb1gt_1,174). happensAt(walking(lb1gt_1),174).
orientation(lb1gt_0,136,175). coords(lb1gt_0,161,46,175).  visible(lb1gt_0,175). happensAt(walking(lb1gt_0),175). orientation(lb1gt_1,151,175). coords(lb1gt_1,102,149,175).  visible(lb1gt_1,175). happensAt(walking(lb1gt_1),175).
orientation(lb1gt_0,136,176). coords(lb1gt_0,161,44,176).  visible(lb1gt_0,176). happensAt(walking(lb1gt_0),176). orientation(lb1gt_1,151,176). coords(lb1gt_1,103,149,176).  visible(lb1gt_1,176). happensAt(walking(lb1gt_1),176).
orientation(lb1gt_0,136,177). coords(lb1gt_0,161,43,177).  visible(lb1gt_0,177). happensAt(walking(lb1gt_0),177). orientation(lb1gt_1,151,177). coords(lb1gt_1,105,149,177).  visible(lb1gt_1,177). happensAt(walking(lb1gt_1),177).
orientation(lb1gt_0,136,178). coords(lb1gt_0,161,41,178).  visible(lb1gt_0,178). happensAt(walking(lb1gt_0),178). orientation(lb1gt_1,151,178). coords(lb1gt_1,106,149,178).  visible(lb1gt_1,178). happensAt(walking(lb1gt_1),178).
orientation(lb1gt_0,136,179). coords(lb1gt_0,161,39,179).  visible(lb1gt_0,179). happensAt(walking(lb1gt_0),179). orientation(lb1gt_1,151,179). coords(lb1gt_1,107,149,179).  visible(lb1gt_1,179). happensAt(walking(lb1gt_1),179).
orientation(lb1gt_0,136,180). coords(lb1gt_0,161,38,180).  visible(lb1gt_0,180). happensAt(walking(lb1gt_0),180). orientation(lb1gt_1,151,180). coords(lb1gt_1,108,149,180).  visible(lb1gt_1,180). happensAt(walking(lb1gt_1),180).
orientation(lb1gt_0,136,181). coords(lb1gt_0,162,38,181).  visible(lb1gt_0,181). happensAt(walking(lb1gt_0),181). orientation(lb1gt_1,151,181). coords(lb1gt_1,109,149,181).  visible(lb1gt_1,181). happensAt(walking(lb1gt_1),181).
orientation(lb1gt_0,136,182). coords(lb1gt_0,162,38,182).  visible(lb1gt_0,182). happensAt(walking(lb1gt_0),182). orientation(lb1gt_1,151,182). coords(lb1gt_1,110,150,182).  visible(lb1gt_1,182). happensAt(walking(lb1gt_1),182).
orientation(lb1gt_0,136,183). coords(lb1gt_0,162,38,183).  visible(lb1gt_0,183). happensAt(walking(lb1gt_0),183). orientation(lb1gt_1,151,183). coords(lb1gt_1,110,150,183).  visible(lb1gt_1,183). happensAt(walking(lb1gt_1),183).
orientation(lb1gt_0,136,184). coords(lb1gt_0,163,38,184).  visible(lb1gt_0,184). happensAt(walking(lb1gt_0),184). orientation(lb1gt_1,151,184). coords(lb1gt_1,111,150,184).  visible(lb1gt_1,184). happensAt(walking(lb1gt_1),184).
orientation(lb1gt_0,136,185). coords(lb1gt_0,163,37,185).  visible(lb1gt_0,185). happensAt(walking(lb1gt_0),185). orientation(lb1gt_1,151,185). coords(lb1gt_1,111,150,185).  visible(lb1gt_1,185). happensAt(walking(lb1gt_1),185).
orientation(lb1gt_0,136,186). coords(lb1gt_0,164,37,186).  visible(lb1gt_0,186). happensAt(walking(lb1gt_0),186). orientation(lb1gt_1,151,186). coords(lb1gt_1,111,150,186).  visible(lb1gt_1,186). happensAt(walking(lb1gt_1),186).
orientation(lb1gt_0,136,187). coords(lb1gt_0,165,35,187).  visible(lb1gt_0,187). happensAt(walking(lb1gt_0),187). orientation(lb1gt_1,151,187). coords(lb1gt_1,111,150,187).  visible(lb1gt_1,187). happensAt(walking(lb1gt_1),187).
orientation(lb1gt_0,136,188). coords(lb1gt_0,165,35,188).  visible(lb1gt_0,188). happensAt(walking(lb1gt_0),188). orientation(lb1gt_1,151,188). coords(lb1gt_1,111,150,188).  visible(lb1gt_1,188). happensAt(walking(lb1gt_1),188).
orientation(lb1gt_0,136,189). coords(lb1gt_0,165,34,189).  visible(lb1gt_0,189). happensAt(walking(lb1gt_0),189). orientation(lb1gt_1,151,189). coords(lb1gt_1,112,150,189).  visible(lb1gt_1,189). happensAt(walking(lb1gt_1),189).
orientation(lb1gt_0,136,190). coords(lb1gt_0,166,34,190).  visible(lb1gt_0,190). happensAt(walking(lb1gt_0),190). orientation(lb1gt_1,151,190). coords(lb1gt_1,112,150,190).  visible(lb1gt_1,190). happensAt(walking(lb1gt_1),190).
orientation(lb1gt_0,136,191). coords(lb1gt_0,166,34,191).  visible(lb1gt_0,191). happensAt(walking(lb1gt_0),191). orientation(lb1gt_1,151,191). coords(lb1gt_1,112,151,191).  visible(lb1gt_1,191). happensAt(walking(lb1gt_1),191).
orientation(lb1gt_0,136,192). coords(lb1gt_0,166,33,192).  visible(lb1gt_0,192). happensAt(walking(lb1gt_0),192). orientation(lb1gt_1,151,192). coords(lb1gt_1,112,151,192).  visible(lb1gt_1,192). happensAt(walking(lb1gt_1),192).
#show holdsAt/2.