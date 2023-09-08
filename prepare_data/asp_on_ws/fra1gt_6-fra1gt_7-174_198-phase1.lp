
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

	orientation(fra1gt_6,101,174). coords(fra1gt_6,292,1,174).  happensAt(appear(fra1gt_6),174). happensAt(walking(fra1gt_6),174). orientation(fra1gt_7,131,174). coords(fra1gt_7,169,18,174).  visible(fra1gt_7,174). happensAt(walking(fra1gt_7),174).
orientation(fra1gt_6,101,175). coords(fra1gt_6,292,2,175).  visible(fra1gt_6,175). happensAt(walking(fra1gt_6),175). orientation(fra1gt_7,131,175). coords(fra1gt_7,169,18,175).  visible(fra1gt_7,175). happensAt(walking(fra1gt_7),175).
orientation(fra1gt_6,101,176). coords(fra1gt_6,290,3,176).  visible(fra1gt_6,176). happensAt(walking(fra1gt_6),176). orientation(fra1gt_7,131,176). coords(fra1gt_7,169,18,176).  visible(fra1gt_7,176). happensAt(walking(fra1gt_7),176).
orientation(fra1gt_6,101,177). coords(fra1gt_6,289,3,177).  visible(fra1gt_6,177). happensAt(walking(fra1gt_6),177). orientation(fra1gt_7,131,177). coords(fra1gt_7,169,18,177).  visible(fra1gt_7,177). happensAt(walking(fra1gt_7),177).
orientation(fra1gt_6,101,178). coords(fra1gt_6,289,4,178).  visible(fra1gt_6,178). happensAt(walking(fra1gt_6),178). orientation(fra1gt_7,131,178). coords(fra1gt_7,169,18,178).  visible(fra1gt_7,178). happensAt(walking(fra1gt_7),178).
orientation(fra1gt_6,101,179). coords(fra1gt_6,289,4,179).  visible(fra1gt_6,179). happensAt(walking(fra1gt_6),179). orientation(fra1gt_7,131,179). coords(fra1gt_7,169,18,179).  visible(fra1gt_7,179). happensAt(walking(fra1gt_7),179).
orientation(fra1gt_6,101,180). coords(fra1gt_6,289,4,180).  visible(fra1gt_6,180). happensAt(walking(fra1gt_6),180). orientation(fra1gt_7,131,180). coords(fra1gt_7,169,19,180).  visible(fra1gt_7,180). happensAt(walking(fra1gt_7),180).
orientation(fra1gt_6,101,181). coords(fra1gt_6,289,4,181).  visible(fra1gt_6,181). happensAt(walking(fra1gt_6),181). orientation(fra1gt_7,131,181). coords(fra1gt_7,169,20,181).  visible(fra1gt_7,181). happensAt(walking(fra1gt_7),181).
orientation(fra1gt_6,101,182). coords(fra1gt_6,289,4,182).  visible(fra1gt_6,182). happensAt(walking(fra1gt_6),182). orientation(fra1gt_7,131,182). coords(fra1gt_7,169,20,182).  visible(fra1gt_7,182). happensAt(walking(fra1gt_7),182).
orientation(fra1gt_6,101,183). coords(fra1gt_6,289,4,183).  visible(fra1gt_6,183). happensAt(walking(fra1gt_6),183). orientation(fra1gt_7,127,183). coords(fra1gt_7,168,21,183).  visible(fra1gt_7,183). happensAt(walking(fra1gt_7),183).
orientation(fra1gt_6,101,184). coords(fra1gt_6,287,4,184).  visible(fra1gt_6,184). happensAt(walking(fra1gt_6),184). orientation(fra1gt_7,123,184). coords(fra1gt_7,168,21,184).  visible(fra1gt_7,184). happensAt(walking(fra1gt_7),184).
orientation(fra1gt_6,101,185). coords(fra1gt_6,286,4,185).  visible(fra1gt_6,185). happensAt(walking(fra1gt_6),185). orientation(fra1gt_7,125,185). coords(fra1gt_7,169,22,185).  visible(fra1gt_7,185). happensAt(walking(fra1gt_7),185).
orientation(fra1gt_6,101,186). coords(fra1gt_6,285,4,186).  visible(fra1gt_6,186). happensAt(walking(fra1gt_6),186). orientation(fra1gt_7,124,186). coords(fra1gt_7,169,21,186).  visible(fra1gt_7,186). happensAt(walking(fra1gt_7),186).
orientation(fra1gt_6,101,187). coords(fra1gt_6,284,5,187).  visible(fra1gt_6,187). happensAt(walking(fra1gt_6),187). orientation(fra1gt_7,125,187). coords(fra1gt_7,169,21,187).  visible(fra1gt_7,187). happensAt(walking(fra1gt_7),187).
orientation(fra1gt_6,101,188). coords(fra1gt_6,284,5,188).  visible(fra1gt_6,188). happensAt(walking(fra1gt_6),188). orientation(fra1gt_7,124,188). coords(fra1gt_7,169,21,188).  visible(fra1gt_7,188). happensAt(walking(fra1gt_7),188).
orientation(fra1gt_6,101,189). coords(fra1gt_6,284,5,189).  visible(fra1gt_6,189). happensAt(walking(fra1gt_6),189). orientation(fra1gt_7,125,189). coords(fra1gt_7,169,22,189).  visible(fra1gt_7,189). happensAt(walking(fra1gt_7),189).
orientation(fra1gt_6,101,190). coords(fra1gt_6,283,5,190).  visible(fra1gt_6,190). happensAt(walking(fra1gt_6),190). orientation(fra1gt_7,126,190). coords(fra1gt_7,169,21,190).  visible(fra1gt_7,190). happensAt(walking(fra1gt_7),190).
orientation(fra1gt_6,101,191). coords(fra1gt_6,282,5,191).  visible(fra1gt_6,191). happensAt(walking(fra1gt_6),191). orientation(fra1gt_7,127,191). coords(fra1gt_7,169,22,191).  visible(fra1gt_7,191). happensAt(walking(fra1gt_7),191).
orientation(fra1gt_6,101,192). coords(fra1gt_6,282,5,192).  visible(fra1gt_6,192). happensAt(walking(fra1gt_6),192). orientation(fra1gt_7,127,192). coords(fra1gt_7,168,22,192).  visible(fra1gt_7,192). happensAt(walking(fra1gt_7),192).
orientation(fra1gt_6,101,193). coords(fra1gt_6,282,5,193).  visible(fra1gt_6,193). happensAt(walking(fra1gt_6),193). orientation(fra1gt_7,127,193). coords(fra1gt_7,168,22,193).  visible(fra1gt_7,193). happensAt(walking(fra1gt_7),193).
orientation(fra1gt_6,101,194). coords(fra1gt_6,281,6,194).  visible(fra1gt_6,194). happensAt(walking(fra1gt_6),194). orientation(fra1gt_7,127,194). coords(fra1gt_7,168,22,194).  visible(fra1gt_7,194). happensAt(walking(fra1gt_7),194).
orientation(fra1gt_6,101,195). coords(fra1gt_6,281,7,195).  visible(fra1gt_6,195). happensAt(walking(fra1gt_6),195). orientation(fra1gt_7,127,195). coords(fra1gt_7,168,23,195).  visible(fra1gt_7,195). happensAt(walking(fra1gt_7),195).
orientation(fra1gt_6,101,196). coords(fra1gt_6,280,7,196).  visible(fra1gt_6,196). happensAt(walking(fra1gt_6),196). orientation(fra1gt_7,127,196). coords(fra1gt_7,168,23,196).  visible(fra1gt_7,196). happensAt(walking(fra1gt_7),196).
orientation(fra1gt_6,101,197). coords(fra1gt_6,280,8,197).  visible(fra1gt_6,197). happensAt(walking(fra1gt_6),197). orientation(fra1gt_7,127,197). coords(fra1gt_7,168,23,197).  visible(fra1gt_7,197). happensAt(walking(fra1gt_7),197).
orientation(fra1gt_6,101,198). coords(fra1gt_6,280,8,198).  visible(fra1gt_6,198). happensAt(walking(fra1gt_6),198). orientation(fra1gt_7,127,198). coords(fra1gt_7,168,23,198).  visible(fra1gt_7,198). happensAt(walking(fra1gt_7),198).
#show holdsAt/2.