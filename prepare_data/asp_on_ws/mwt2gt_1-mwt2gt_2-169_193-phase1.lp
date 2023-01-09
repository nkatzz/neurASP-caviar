
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

	orientation(mwt2gt_1,137,169). coords(mwt2gt_1,166,35,169).  visible(mwt2gt_1,169). happensAt(walking(mwt2gt_1),169). orientation(mwt2gt_2,101,169). coords(mwt2gt_2,270,283,169).  visible(mwt2gt_2,169). happensAt(walking(mwt2gt_2),169).
orientation(mwt2gt_1,137,170). coords(mwt2gt_1,166,35,170).  visible(mwt2gt_1,170). happensAt(walking(mwt2gt_1),170). orientation(mwt2gt_2,101,170). coords(mwt2gt_2,270,283,170).  visible(mwt2gt_2,170). happensAt(walking(mwt2gt_2),170).
orientation(mwt2gt_1,137,171). coords(mwt2gt_1,166,35,171).  visible(mwt2gt_1,171). happensAt(walking(mwt2gt_1),171). orientation(mwt2gt_2,101,171). coords(mwt2gt_2,269,283,171).  visible(mwt2gt_2,171). happensAt(walking(mwt2gt_2),171).
orientation(mwt2gt_1,137,172). coords(mwt2gt_1,166,35,172).  visible(mwt2gt_1,172). happensAt(walking(mwt2gt_1),172). orientation(mwt2gt_2,101,172). coords(mwt2gt_2,269,283,172).  visible(mwt2gt_2,172). happensAt(walking(mwt2gt_2),172).
orientation(mwt2gt_1,137,173). coords(mwt2gt_1,166,37,173).  visible(mwt2gt_1,173). happensAt(walking(mwt2gt_1),173). orientation(mwt2gt_2,101,173). coords(mwt2gt_2,269,283,173).  visible(mwt2gt_2,173). happensAt(walking(mwt2gt_2),173).
orientation(mwt2gt_1,137,174). coords(mwt2gt_1,166,37,174).  visible(mwt2gt_1,174). happensAt(walking(mwt2gt_1),174). orientation(mwt2gt_2,101,174). coords(mwt2gt_2,269,283,174).  visible(mwt2gt_2,174). happensAt(walking(mwt2gt_2),174).
orientation(mwt2gt_1,137,175). coords(mwt2gt_1,166,37,175).  visible(mwt2gt_1,175). happensAt(walking(mwt2gt_1),175). orientation(mwt2gt_2,101,175). coords(mwt2gt_2,269,283,175).  visible(mwt2gt_2,175). happensAt(walking(mwt2gt_2),175).
orientation(mwt2gt_1,137,176). coords(mwt2gt_1,165,37,176).  visible(mwt2gt_1,176). happensAt(walking(mwt2gt_1),176). orientation(mwt2gt_2,81,176). coords(mwt2gt_2,269,283,176).  visible(mwt2gt_2,176). happensAt(walking(mwt2gt_2),176).
orientation(mwt2gt_1,137,177). coords(mwt2gt_1,165,37,177).  visible(mwt2gt_1,177). happensAt(walking(mwt2gt_1),177). orientation(mwt2gt_2,67,177). coords(mwt2gt_2,263,283,177).  visible(mwt2gt_2,177). happensAt(walking(mwt2gt_2),177).
orientation(mwt2gt_1,137,178). coords(mwt2gt_1,164,37,178).  visible(mwt2gt_1,178). happensAt(walking(mwt2gt_1),178). orientation(mwt2gt_2,64,178). coords(mwt2gt_2,262,283,178).  visible(mwt2gt_2,178). happensAt(walking(mwt2gt_2),178).
orientation(mwt2gt_1,137,179). coords(mwt2gt_1,163,37,179).  visible(mwt2gt_1,179). happensAt(walking(mwt2gt_1),179). orientation(mwt2gt_2,59,179). coords(mwt2gt_2,261,280,179).  visible(mwt2gt_2,179). happensAt(walking(mwt2gt_2),179).
orientation(mwt2gt_1,137,180). coords(mwt2gt_1,162,37,180).  visible(mwt2gt_1,180). happensAt(walking(mwt2gt_1),180). orientation(mwt2gt_2,59,180). coords(mwt2gt_2,260,277,180).  visible(mwt2gt_2,180). happensAt(walking(mwt2gt_2),180).
orientation(mwt2gt_1,137,181). coords(mwt2gt_1,162,37,181).  visible(mwt2gt_1,181). happensAt(walking(mwt2gt_1),181). orientation(mwt2gt_2,59,181). coords(mwt2gt_2,260,277,181).  visible(mwt2gt_2,181). happensAt(walking(mwt2gt_2),181).
orientation(mwt2gt_1,137,182). coords(mwt2gt_1,162,37,182).  visible(mwt2gt_1,182). happensAt(walking(mwt2gt_1),182). orientation(mwt2gt_2,60,182). coords(mwt2gt_2,255,272,182).  visible(mwt2gt_2,182). happensAt(walking(mwt2gt_2),182).
orientation(mwt2gt_1,137,183). coords(mwt2gt_1,162,37,183).  visible(mwt2gt_1,183). happensAt(walking(mwt2gt_1),183). orientation(mwt2gt_2,42,183). coords(mwt2gt_2,254,272,183).  visible(mwt2gt_2,183). happensAt(walking(mwt2gt_2),183).
orientation(mwt2gt_1,137,184). coords(mwt2gt_1,162,37,184).  visible(mwt2gt_1,184). happensAt(walking(mwt2gt_1),184). orientation(mwt2gt_2,46,184). coords(mwt2gt_2,254,271,184).  visible(mwt2gt_2,184). happensAt(walking(mwt2gt_2),184).
orientation(mwt2gt_1,137,185). coords(mwt2gt_1,161,40,185).  visible(mwt2gt_1,185). happensAt(walking(mwt2gt_1),185). orientation(mwt2gt_2,34,185). coords(mwt2gt_2,253,271,185).  visible(mwt2gt_2,185). happensAt(walking(mwt2gt_2),185).
orientation(mwt2gt_1,137,186). coords(mwt2gt_1,161,40,186).  visible(mwt2gt_1,186). happensAt(walking(mwt2gt_1),186). orientation(mwt2gt_2,29,186). coords(mwt2gt_2,253,271,186).  visible(mwt2gt_2,186). happensAt(walking(mwt2gt_2),186).
orientation(mwt2gt_1,137,187). coords(mwt2gt_1,160,40,187).  visible(mwt2gt_1,187). happensAt(walking(mwt2gt_1),187). orientation(mwt2gt_2,29,187). coords(mwt2gt_2,252,269,187).  visible(mwt2gt_2,187). happensAt(walking(mwt2gt_2),187).
orientation(mwt2gt_1,137,188). coords(mwt2gt_1,160,40,188).  visible(mwt2gt_1,188). happensAt(walking(mwt2gt_1),188). orientation(mwt2gt_2,29,188). coords(mwt2gt_2,251,267,188).  visible(mwt2gt_2,188). happensAt(walking(mwt2gt_2),188).
orientation(mwt2gt_1,137,189). coords(mwt2gt_1,160,41,189).  visible(mwt2gt_1,189). happensAt(walking(mwt2gt_1),189). orientation(mwt2gt_2,29,189). coords(mwt2gt_2,250,266,189).  visible(mwt2gt_2,189). happensAt(walking(mwt2gt_2),189).
orientation(mwt2gt_1,137,190). coords(mwt2gt_1,160,41,190).  visible(mwt2gt_1,190). happensAt(walking(mwt2gt_1),190). orientation(mwt2gt_2,25,190). coords(mwt2gt_2,249,265,190).  visible(mwt2gt_2,190). happensAt(walking(mwt2gt_2),190).
orientation(mwt2gt_1,141,191). coords(mwt2gt_1,158,41,191).  visible(mwt2gt_1,191). happensAt(walking(mwt2gt_1),191). orientation(mwt2gt_2,25,191). coords(mwt2gt_2,249,263,191).  visible(mwt2gt_2,191). happensAt(walking(mwt2gt_2),191).
orientation(mwt2gt_1,141,192). coords(mwt2gt_1,158,41,192).  visible(mwt2gt_1,192). happensAt(walking(mwt2gt_1),192). orientation(mwt2gt_2,25,192). coords(mwt2gt_2,247,261,192).  visible(mwt2gt_2,192). happensAt(walking(mwt2gt_2),192).
orientation(mwt2gt_1,141,193). coords(mwt2gt_1,156,41,193).  visible(mwt2gt_1,193). happensAt(walking(mwt2gt_1),193). orientation(mwt2gt_2,25,193). coords(mwt2gt_2,247,257,193).  visible(mwt2gt_2,193). happensAt(walking(mwt2gt_2),193).
#show holdsAt/2.