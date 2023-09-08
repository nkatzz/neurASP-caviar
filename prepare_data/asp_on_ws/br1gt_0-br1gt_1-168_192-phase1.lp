
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

	orientation(br1gt_0,44,168). coords(br1gt_0,273,243,168).  visible(br1gt_0,168). happensAt(walking(br1gt_0),168). orientation(br1gt_1,164,168). coords(br1gt_1,43,156,168).  visible(br1gt_1,168). happensAt(walking(br1gt_1),168).
orientation(br1gt_0,33,169). coords(br1gt_0,277,245,169).  visible(br1gt_0,169). happensAt(walking(br1gt_0),169). orientation(br1gt_1,164,169). coords(br1gt_1,42,156,169).  visible(br1gt_1,169). happensAt(walking(br1gt_1),169).
orientation(br1gt_0,33,170). coords(br1gt_0,281,246,170).  visible(br1gt_0,170). happensAt(walking(br1gt_0),170). orientation(br1gt_1,164,170). coords(br1gt_1,41,156,170).  visible(br1gt_1,170). happensAt(walking(br1gt_1),170).
orientation(br1gt_0,33,171). coords(br1gt_0,284,248,171).  visible(br1gt_0,171). happensAt(walking(br1gt_0),171). orientation(br1gt_1,164,171). coords(br1gt_1,41,156,171).  visible(br1gt_1,171). happensAt(walking(br1gt_1),171).
orientation(br1gt_0,33,172). coords(br1gt_0,286,249,172).  visible(br1gt_0,172). happensAt(walking(br1gt_0),172). orientation(br1gt_1,164,172). coords(br1gt_1,40,155,172).  visible(br1gt_1,172). happensAt(walking(br1gt_1),172).
orientation(br1gt_0,33,173). coords(br1gt_0,289,251,173).  visible(br1gt_0,173). happensAt(walking(br1gt_0),173). orientation(br1gt_1,164,173). coords(br1gt_1,40,155,173).  visible(br1gt_1,173). happensAt(walking(br1gt_1),173).
orientation(br1gt_0,33,174). coords(br1gt_0,291,253,174).  visible(br1gt_0,174). happensAt(walking(br1gt_0),174). orientation(br1gt_1,164,174). coords(br1gt_1,40,155,174).  visible(br1gt_1,174). happensAt(walking(br1gt_1),174).
orientation(br1gt_0,33,175). coords(br1gt_0,292,256,175).  visible(br1gt_0,175). happensAt(walking(br1gt_0),175). orientation(br1gt_1,164,175). coords(br1gt_1,40,155,175).  visible(br1gt_1,175). happensAt(walking(br1gt_1),175).
orientation(br1gt_0,39,176). coords(br1gt_0,293,258,176).  visible(br1gt_0,176). happensAt(walking(br1gt_0),176). orientation(br1gt_1,164,176). coords(br1gt_1,39,155,176).  visible(br1gt_1,176). happensAt(walking(br1gt_1),176).
orientation(br1gt_0,39,177). coords(br1gt_0,294,260,177).  visible(br1gt_0,177). happensAt(walking(br1gt_0),177). orientation(br1gt_1,164,177). coords(br1gt_1,39,155,177).  visible(br1gt_1,177). happensAt(walking(br1gt_1),177).
orientation(br1gt_0,39,178). coords(br1gt_0,295,263,178).  visible(br1gt_0,178). happensAt(walking(br1gt_0),178). orientation(br1gt_1,164,178). coords(br1gt_1,39,155,178).  visible(br1gt_1,178). happensAt(walking(br1gt_1),178).
orientation(br1gt_0,50,179). coords(br1gt_0,296,266,179).  visible(br1gt_0,179). happensAt(walking(br1gt_0),179). orientation(br1gt_1,164,179). coords(br1gt_1,39,155,179).  visible(br1gt_1,179). happensAt(walking(br1gt_1),179).
orientation(br1gt_0,50,180). coords(br1gt_0,298,268,180).  visible(br1gt_0,180). happensAt(walking(br1gt_0),180). orientation(br1gt_1,164,180). coords(br1gt_1,39,155,180).  visible(br1gt_1,180). happensAt(walking(br1gt_1),180).
orientation(br1gt_0,50,181). coords(br1gt_0,299,270,181).  visible(br1gt_0,181). happensAt(walking(br1gt_0),181). orientation(br1gt_1,164,181). coords(br1gt_1,39,155,181).  visible(br1gt_1,181). happensAt(walking(br1gt_1),181).
orientation(br1gt_0,50,182). coords(br1gt_0,299,270,182).  visible(br1gt_0,182). happensAt(walking(br1gt_0),182). orientation(br1gt_1,164,182). coords(br1gt_1,39,155,182).  visible(br1gt_1,182). happensAt(walking(br1gt_1),182).
orientation(br1gt_0,50,183). coords(br1gt_0,300,270,183).  visible(br1gt_0,183). happensAt(walking(br1gt_0),183). orientation(br1gt_1,164,183). coords(br1gt_1,37,155,183).  visible(br1gt_1,183). happensAt(walking(br1gt_1),183).
orientation(br1gt_0,50,184). coords(br1gt_0,300,270,184).  visible(br1gt_0,184). happensAt(walking(br1gt_0),184). orientation(br1gt_1,164,184). coords(br1gt_1,37,155,184).  visible(br1gt_1,184). happensAt(walking(br1gt_1),184).
orientation(br1gt_0,50,185). coords(br1gt_0,300,270,185).  visible(br1gt_0,185). happensAt(walking(br1gt_0),185). orientation(br1gt_1,164,185). coords(br1gt_1,37,155,185).  visible(br1gt_1,185). happensAt(walking(br1gt_1),185).
orientation(br1gt_0,50,186). coords(br1gt_0,299,270,186).  visible(br1gt_0,186). happensAt(walking(br1gt_0),186). orientation(br1gt_1,164,186). coords(br1gt_1,37,155,186).  visible(br1gt_1,186). happensAt(walking(br1gt_1),186).
orientation(br1gt_0,50,187). coords(br1gt_0,300,271,187).  visible(br1gt_0,187). happensAt(walking(br1gt_0),187). orientation(br1gt_1,164,187). coords(br1gt_1,35,155,187).  visible(br1gt_1,187). happensAt(walking(br1gt_1),187).
orientation(br1gt_0,50,188). coords(br1gt_0,300,273,188).  visible(br1gt_0,188). happensAt(walking(br1gt_0),188). orientation(br1gt_1,164,188). coords(br1gt_1,35,155,188).  visible(br1gt_1,188). happensAt(walking(br1gt_1),188).
orientation(br1gt_0,50,189). coords(br1gt_0,300,275,189).  visible(br1gt_0,189). happensAt(walking(br1gt_0),189). orientation(br1gt_1,164,189). coords(br1gt_1,35,156,189).  visible(br1gt_1,189). happensAt(walking(br1gt_1),189).
orientation(br1gt_0,47,190). coords(br1gt_0,300,275,190).  visible(br1gt_0,190). happensAt(walking(br1gt_0),190). orientation(br1gt_1,164,190). coords(br1gt_1,35,156,190).  visible(br1gt_1,190). happensAt(walking(br1gt_1),190).
orientation(br1gt_0,47,191). coords(br1gt_0,300,277,191).  visible(br1gt_0,191). happensAt(walking(br1gt_0),191). orientation(br1gt_1,164,191). coords(br1gt_1,35,156,191).  visible(br1gt_1,191). happensAt(walking(br1gt_1),191).
orientation(br1gt_0,47,192). coords(br1gt_0,301,278,192).  visible(br1gt_0,192). happensAt(walking(br1gt_0),192). orientation(br1gt_1,164,192). coords(br1gt_1,35,156,192).  visible(br1gt_1,192). happensAt(walking(br1gt_1),192).
#show holdsAt/2.