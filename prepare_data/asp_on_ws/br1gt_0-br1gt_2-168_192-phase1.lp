
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

	orientation(br1gt_0,44,168). coords(br1gt_0,273,243,168).  visible(br1gt_0,168). happensAt(walking(br1gt_0),168). orientation(br1gt_2,171,168). coords(br1gt_2,86,207,168).  visible(br1gt_2,168). happensAt(inactive(br1gt_2),168).
orientation(br1gt_0,33,169). coords(br1gt_0,277,245,169).  visible(br1gt_0,169). happensAt(walking(br1gt_0),169). orientation(br1gt_2,171,169). coords(br1gt_2,86,207,169).  visible(br1gt_2,169). happensAt(inactive(br1gt_2),169).
orientation(br1gt_0,33,170). coords(br1gt_0,281,246,170).  visible(br1gt_0,170). happensAt(walking(br1gt_0),170). orientation(br1gt_2,171,170). coords(br1gt_2,86,207,170).  visible(br1gt_2,170). happensAt(inactive(br1gt_2),170).
orientation(br1gt_0,33,171). coords(br1gt_0,284,248,171).  visible(br1gt_0,171). happensAt(walking(br1gt_0),171). orientation(br1gt_2,171,171). coords(br1gt_2,86,207,171).  visible(br1gt_2,171). happensAt(inactive(br1gt_2),171).
orientation(br1gt_0,33,172). coords(br1gt_0,286,249,172).  visible(br1gt_0,172). happensAt(walking(br1gt_0),172). orientation(br1gt_2,171,172). coords(br1gt_2,86,207,172).  visible(br1gt_2,172). happensAt(inactive(br1gt_2),172).
orientation(br1gt_0,33,173). coords(br1gt_0,289,251,173).  visible(br1gt_0,173). happensAt(walking(br1gt_0),173). orientation(br1gt_2,171,173). coords(br1gt_2,86,207,173).  visible(br1gt_2,173). happensAt(inactive(br1gt_2),173).
orientation(br1gt_0,33,174). coords(br1gt_0,291,253,174).  visible(br1gt_0,174). happensAt(walking(br1gt_0),174). orientation(br1gt_2,171,174). coords(br1gt_2,86,207,174).  visible(br1gt_2,174). happensAt(inactive(br1gt_2),174).
orientation(br1gt_0,33,175). coords(br1gt_0,292,256,175).  visible(br1gt_0,175). happensAt(walking(br1gt_0),175). orientation(br1gt_2,171,175). coords(br1gt_2,86,207,175).  visible(br1gt_2,175). happensAt(inactive(br1gt_2),175).
orientation(br1gt_0,39,176). coords(br1gt_0,293,258,176).  visible(br1gt_0,176). happensAt(walking(br1gt_0),176). orientation(br1gt_2,171,176). coords(br1gt_2,86,207,176).  visible(br1gt_2,176). happensAt(inactive(br1gt_2),176).
orientation(br1gt_0,39,177). coords(br1gt_0,294,260,177).  visible(br1gt_0,177). happensAt(walking(br1gt_0),177). orientation(br1gt_2,171,177). coords(br1gt_2,86,207,177).  visible(br1gt_2,177). happensAt(inactive(br1gt_2),177).
orientation(br1gt_0,39,178). coords(br1gt_0,295,263,178).  visible(br1gt_0,178). happensAt(walking(br1gt_0),178). orientation(br1gt_2,171,178). coords(br1gt_2,86,207,178).  visible(br1gt_2,178). happensAt(inactive(br1gt_2),178).
orientation(br1gt_0,50,179). coords(br1gt_0,296,266,179).  visible(br1gt_0,179). happensAt(walking(br1gt_0),179). orientation(br1gt_2,171,179). coords(br1gt_2,86,207,179).  visible(br1gt_2,179). happensAt(inactive(br1gt_2),179).
orientation(br1gt_0,50,180). coords(br1gt_0,298,268,180).  visible(br1gt_0,180). happensAt(walking(br1gt_0),180). orientation(br1gt_2,171,180). coords(br1gt_2,86,207,180).  visible(br1gt_2,180). happensAt(inactive(br1gt_2),180).
orientation(br1gt_0,50,181). coords(br1gt_0,299,270,181).  visible(br1gt_0,181). happensAt(walking(br1gt_0),181). orientation(br1gt_2,171,181). coords(br1gt_2,86,207,181).  visible(br1gt_2,181). happensAt(inactive(br1gt_2),181).
orientation(br1gt_0,50,182). coords(br1gt_0,299,270,182).  visible(br1gt_0,182). happensAt(walking(br1gt_0),182). orientation(br1gt_2,171,182). coords(br1gt_2,86,207,182).  visible(br1gt_2,182). happensAt(inactive(br1gt_2),182).
orientation(br1gt_0,50,183). coords(br1gt_0,300,270,183).  visible(br1gt_0,183). happensAt(walking(br1gt_0),183). orientation(br1gt_2,171,183). coords(br1gt_2,86,207,183).  visible(br1gt_2,183). happensAt(inactive(br1gt_2),183).
orientation(br1gt_0,50,184). coords(br1gt_0,300,270,184).  visible(br1gt_0,184). happensAt(walking(br1gt_0),184). orientation(br1gt_2,171,184). coords(br1gt_2,86,207,184).  visible(br1gt_2,184). happensAt(inactive(br1gt_2),184).
orientation(br1gt_0,50,185). coords(br1gt_0,300,270,185).  visible(br1gt_0,185). happensAt(walking(br1gt_0),185). orientation(br1gt_2,171,185). coords(br1gt_2,86,207,185).  visible(br1gt_2,185). happensAt(inactive(br1gt_2),185).
orientation(br1gt_0,50,186). coords(br1gt_0,299,270,186).  visible(br1gt_0,186). happensAt(walking(br1gt_0),186). orientation(br1gt_2,171,186). coords(br1gt_2,86,207,186).  visible(br1gt_2,186). happensAt(inactive(br1gt_2),186).
orientation(br1gt_0,50,187). coords(br1gt_0,300,271,187).  visible(br1gt_0,187). happensAt(walking(br1gt_0),187). orientation(br1gt_2,171,187). coords(br1gt_2,86,207,187).  visible(br1gt_2,187). happensAt(inactive(br1gt_2),187).
orientation(br1gt_0,50,188). coords(br1gt_0,300,273,188).  visible(br1gt_0,188). happensAt(walking(br1gt_0),188). orientation(br1gt_2,171,188). coords(br1gt_2,86,207,188).  visible(br1gt_2,188). happensAt(inactive(br1gt_2),188).
orientation(br1gt_0,50,189). coords(br1gt_0,300,275,189).  visible(br1gt_0,189). happensAt(walking(br1gt_0),189). orientation(br1gt_2,171,189). coords(br1gt_2,86,207,189).  visible(br1gt_2,189). happensAt(inactive(br1gt_2),189).
orientation(br1gt_0,47,190). coords(br1gt_0,300,275,190).  visible(br1gt_0,190). happensAt(walking(br1gt_0),190). orientation(br1gt_2,171,190). coords(br1gt_2,86,207,190).  visible(br1gt_2,190). happensAt(inactive(br1gt_2),190).
orientation(br1gt_0,47,191). coords(br1gt_0,300,277,191).  visible(br1gt_0,191). happensAt(walking(br1gt_0),191). orientation(br1gt_2,171,191). coords(br1gt_2,86,207,191).  visible(br1gt_2,191). happensAt(inactive(br1gt_2),191).
orientation(br1gt_0,47,192). coords(br1gt_0,301,278,192).  visible(br1gt_0,192). happensAt(walking(br1gt_0),192). orientation(br1gt_2,171,192). coords(br1gt_2,86,207,192).  visible(br1gt_2,192). happensAt(inactive(br1gt_2),192).
#show holdsAt/2.