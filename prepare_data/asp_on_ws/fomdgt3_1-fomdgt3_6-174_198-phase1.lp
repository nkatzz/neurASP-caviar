
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

	orientation(fomdgt3_1,150,174). coords(fomdgt3_1,88,105,174).  visible(fomdgt3_1,174). happensAt(walking(fomdgt3_1),174). orientation(fomdgt3_6,130,174). coords(fomdgt3_6,230,45,174).  visible(fomdgt3_6,174). happensAt(walking(fomdgt3_6),174).
orientation(fomdgt3_1,150,175). coords(fomdgt3_1,89,105,175).  visible(fomdgt3_1,175). happensAt(walking(fomdgt3_1),175). orientation(fomdgt3_6,130,175). coords(fomdgt3_6,229,46,175).  visible(fomdgt3_6,175). happensAt(walking(fomdgt3_6),175).
orientation(fomdgt3_1,150,176). coords(fomdgt3_1,89,105,176).  visible(fomdgt3_1,176). happensAt(walking(fomdgt3_1),176). orientation(fomdgt3_6,130,176). coords(fomdgt3_6,229,46,176).  visible(fomdgt3_6,176). happensAt(walking(fomdgt3_6),176).
orientation(fomdgt3_1,150,177). coords(fomdgt3_1,89,105,177).  visible(fomdgt3_1,177). happensAt(walking(fomdgt3_1),177). orientation(fomdgt3_6,130,177). coords(fomdgt3_6,228,49,177).  visible(fomdgt3_6,177). happensAt(walking(fomdgt3_6),177).
orientation(fomdgt3_1,150,178). coords(fomdgt3_1,88,105,178).  visible(fomdgt3_1,178). happensAt(walking(fomdgt3_1),178). orientation(fomdgt3_6,130,178). coords(fomdgt3_6,228,49,178).  visible(fomdgt3_6,178). happensAt(walking(fomdgt3_6),178).
orientation(fomdgt3_1,150,179). coords(fomdgt3_1,89,105,179).  visible(fomdgt3_1,179). happensAt(walking(fomdgt3_1),179). orientation(fomdgt3_6,130,179). coords(fomdgt3_6,227,50,179).  visible(fomdgt3_6,179). happensAt(walking(fomdgt3_6),179).
orientation(fomdgt3_1,150,180). coords(fomdgt3_1,87,105,180).  visible(fomdgt3_1,180). happensAt(walking(fomdgt3_1),180). orientation(fomdgt3_6,130,180). coords(fomdgt3_6,227,52,180).  visible(fomdgt3_6,180). happensAt(walking(fomdgt3_6),180).
orientation(fomdgt3_1,150,181). coords(fomdgt3_1,86,103,181).  visible(fomdgt3_1,181). happensAt(walking(fomdgt3_1),181). orientation(fomdgt3_6,130,181). coords(fomdgt3_6,227,52,181).  visible(fomdgt3_6,181). happensAt(walking(fomdgt3_6),181).
orientation(fomdgt3_1,150,182). coords(fomdgt3_1,86,103,182).  visible(fomdgt3_1,182). happensAt(walking(fomdgt3_1),182). orientation(fomdgt3_6,130,182). coords(fomdgt3_6,223,52,182).  visible(fomdgt3_6,182). happensAt(walking(fomdgt3_6),182).
orientation(fomdgt3_1,150,183). coords(fomdgt3_1,86,103,183).  visible(fomdgt3_1,183). happensAt(walking(fomdgt3_1),183). orientation(fomdgt3_6,130,183). coords(fomdgt3_6,223,52,183).  visible(fomdgt3_6,183). happensAt(walking(fomdgt3_6),183).
orientation(fomdgt3_1,150,184). coords(fomdgt3_1,83,102,184).  visible(fomdgt3_1,184). happensAt(walking(fomdgt3_1),184). orientation(fomdgt3_6,130,184). coords(fomdgt3_6,223,56,184).  visible(fomdgt3_6,184). happensAt(walking(fomdgt3_6),184).
orientation(fomdgt3_1,150,185). coords(fomdgt3_1,83,102,185).  visible(fomdgt3_1,185). happensAt(walking(fomdgt3_1),185). orientation(fomdgt3_6,130,185). coords(fomdgt3_6,220,60,185).  visible(fomdgt3_6,185). happensAt(walking(fomdgt3_6),185).
orientation(fomdgt3_1,150,186). coords(fomdgt3_1,82,102,186).  visible(fomdgt3_1,186). happensAt(walking(fomdgt3_1),186). orientation(fomdgt3_6,130,186). coords(fomdgt3_6,220,63,186).  visible(fomdgt3_6,186). happensAt(walking(fomdgt3_6),186).
orientation(fomdgt3_1,150,187). coords(fomdgt3_1,82,102,187).  visible(fomdgt3_1,187). happensAt(walking(fomdgt3_1),187). orientation(fomdgt3_6,130,187). coords(fomdgt3_6,220,64,187).  visible(fomdgt3_6,187). happensAt(walking(fomdgt3_6),187).
orientation(fomdgt3_1,150,188). coords(fomdgt3_1,82,102,188).  visible(fomdgt3_1,188). happensAt(walking(fomdgt3_1),188). orientation(fomdgt3_6,130,188). coords(fomdgt3_6,220,66,188).  visible(fomdgt3_6,188). happensAt(walking(fomdgt3_6),188).
orientation(fomdgt3_1,150,189). coords(fomdgt3_1,82,102,189).  visible(fomdgt3_1,189). happensAt(walking(fomdgt3_1),189). orientation(fomdgt3_6,130,189). coords(fomdgt3_6,220,68,189).  visible(fomdgt3_6,189). happensAt(walking(fomdgt3_6),189).
orientation(fomdgt3_1,150,190). coords(fomdgt3_1,82,102,190).  visible(fomdgt3_1,190). happensAt(walking(fomdgt3_1),190). orientation(fomdgt3_6,130,190). coords(fomdgt3_6,219,69,190).  visible(fomdgt3_6,190). happensAt(walking(fomdgt3_6),190).
orientation(fomdgt3_1,150,191). coords(fomdgt3_1,82,102,191).  visible(fomdgt3_1,191). happensAt(walking(fomdgt3_1),191). orientation(fomdgt3_6,130,191). coords(fomdgt3_6,218,70,191).  visible(fomdgt3_6,191). happensAt(walking(fomdgt3_6),191).
orientation(fomdgt3_1,150,192). coords(fomdgt3_1,82,102,192).  visible(fomdgt3_1,192). happensAt(walking(fomdgt3_1),192). orientation(fomdgt3_6,130,192). coords(fomdgt3_6,218,71,192).  visible(fomdgt3_6,192). happensAt(walking(fomdgt3_6),192).
orientation(fomdgt3_1,150,193). coords(fomdgt3_1,82,102,193).  visible(fomdgt3_1,193). happensAt(walking(fomdgt3_1),193). orientation(fomdgt3_6,130,193). coords(fomdgt3_6,218,70,193).  visible(fomdgt3_6,193). happensAt(walking(fomdgt3_6),193).
orientation(fomdgt3_1,150,194). coords(fomdgt3_1,82,102,194).  visible(fomdgt3_1,194). happensAt(walking(fomdgt3_1),194). orientation(fomdgt3_6,130,194). coords(fomdgt3_6,217,72,194).  visible(fomdgt3_6,194). happensAt(walking(fomdgt3_6),194).
orientation(fomdgt3_1,150,195). coords(fomdgt3_1,79,99,195).  visible(fomdgt3_1,195). happensAt(walking(fomdgt3_1),195). orientation(fomdgt3_6,130,195). coords(fomdgt3_6,216,72,195).  visible(fomdgt3_6,195). happensAt(walking(fomdgt3_6),195).
orientation(fomdgt3_1,150,196). coords(fomdgt3_1,79,99,196).  visible(fomdgt3_1,196). happensAt(walking(fomdgt3_1),196). orientation(fomdgt3_6,130,196). coords(fomdgt3_6,215,72,196).  visible(fomdgt3_6,196). happensAt(walking(fomdgt3_6),196).
orientation(fomdgt3_1,150,197). coords(fomdgt3_1,79,99,197).  visible(fomdgt3_1,197). happensAt(walking(fomdgt3_1),197). orientation(fomdgt3_6,130,197). coords(fomdgt3_6,212,75,197).  visible(fomdgt3_6,197). happensAt(walking(fomdgt3_6),197).
orientation(fomdgt3_1,150,198). coords(fomdgt3_1,79,99,198).  visible(fomdgt3_1,198). happensAt(walking(fomdgt3_1),198). orientation(fomdgt3_6,130,198). coords(fomdgt3_6,211,79,198).  visible(fomdgt3_6,198). happensAt(walking(fomdgt3_6),198).
#show holdsAt/2.