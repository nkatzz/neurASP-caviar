
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

	orientation(br1gt_1,164,180). coords(br1gt_1,39,155,180).  visible(br1gt_1,180). happensAt(walking(br1gt_1),180). orientation(br1gt_2,171,180). coords(br1gt_2,86,207,180).  visible(br1gt_2,180). happensAt(inactive(br1gt_2),180).
orientation(br1gt_1,164,181). coords(br1gt_1,39,155,181).  visible(br1gt_1,181). happensAt(walking(br1gt_1),181). orientation(br1gt_2,171,181). coords(br1gt_2,86,207,181).  visible(br1gt_2,181). happensAt(inactive(br1gt_2),181).
orientation(br1gt_1,164,182). coords(br1gt_1,39,155,182).  visible(br1gt_1,182). happensAt(walking(br1gt_1),182). orientation(br1gt_2,171,182). coords(br1gt_2,86,207,182).  visible(br1gt_2,182). happensAt(inactive(br1gt_2),182).
orientation(br1gt_1,164,183). coords(br1gt_1,37,155,183).  visible(br1gt_1,183). happensAt(walking(br1gt_1),183). orientation(br1gt_2,171,183). coords(br1gt_2,86,207,183).  visible(br1gt_2,183). happensAt(inactive(br1gt_2),183).
orientation(br1gt_1,164,184). coords(br1gt_1,37,155,184).  visible(br1gt_1,184). happensAt(walking(br1gt_1),184). orientation(br1gt_2,171,184). coords(br1gt_2,86,207,184).  visible(br1gt_2,184). happensAt(inactive(br1gt_2),184).
orientation(br1gt_1,164,185). coords(br1gt_1,37,155,185).  visible(br1gt_1,185). happensAt(walking(br1gt_1),185). orientation(br1gt_2,171,185). coords(br1gt_2,86,207,185).  visible(br1gt_2,185). happensAt(inactive(br1gt_2),185).
orientation(br1gt_1,164,186). coords(br1gt_1,37,155,186).  visible(br1gt_1,186). happensAt(walking(br1gt_1),186). orientation(br1gt_2,171,186). coords(br1gt_2,86,207,186).  visible(br1gt_2,186). happensAt(inactive(br1gt_2),186).
orientation(br1gt_1,164,187). coords(br1gt_1,35,155,187).  visible(br1gt_1,187). happensAt(walking(br1gt_1),187). orientation(br1gt_2,171,187). coords(br1gt_2,86,207,187).  visible(br1gt_2,187). happensAt(inactive(br1gt_2),187).
orientation(br1gt_1,164,188). coords(br1gt_1,35,155,188).  visible(br1gt_1,188). happensAt(walking(br1gt_1),188). orientation(br1gt_2,171,188). coords(br1gt_2,86,207,188).  visible(br1gt_2,188). happensAt(inactive(br1gt_2),188).
orientation(br1gt_1,164,189). coords(br1gt_1,35,156,189).  visible(br1gt_1,189). happensAt(walking(br1gt_1),189). orientation(br1gt_2,171,189). coords(br1gt_2,86,207,189).  visible(br1gt_2,189). happensAt(inactive(br1gt_2),189).
orientation(br1gt_1,164,190). coords(br1gt_1,35,156,190).  visible(br1gt_1,190). happensAt(walking(br1gt_1),190). orientation(br1gt_2,171,190). coords(br1gt_2,86,207,190).  visible(br1gt_2,190). happensAt(inactive(br1gt_2),190).
orientation(br1gt_1,164,191). coords(br1gt_1,35,156,191).  visible(br1gt_1,191). happensAt(walking(br1gt_1),191). orientation(br1gt_2,171,191). coords(br1gt_2,86,207,191).  visible(br1gt_2,191). happensAt(inactive(br1gt_2),191).
orientation(br1gt_1,164,192). coords(br1gt_1,35,156,192).  visible(br1gt_1,192). happensAt(walking(br1gt_1),192). orientation(br1gt_2,171,192). coords(br1gt_2,86,207,192).  visible(br1gt_2,192). happensAt(inactive(br1gt_2),192).
orientation(br1gt_1,164,193). coords(br1gt_1,35,156,193).  visible(br1gt_1,193). happensAt(walking(br1gt_1),193). orientation(br1gt_2,171,193). coords(br1gt_2,86,207,193).  visible(br1gt_2,193). happensAt(inactive(br1gt_2),193).
orientation(br1gt_1,164,194). coords(br1gt_1,35,156,194).  visible(br1gt_1,194). happensAt(walking(br1gt_1),194). orientation(br1gt_2,171,194). coords(br1gt_2,86,207,194).  visible(br1gt_2,194). happensAt(inactive(br1gt_2),194).
orientation(br1gt_1,164,195). coords(br1gt_1,34,156,195).  visible(br1gt_1,195). happensAt(walking(br1gt_1),195). orientation(br1gt_2,171,195). coords(br1gt_2,86,207,195).  visible(br1gt_2,195). happensAt(inactive(br1gt_2),195).
orientation(br1gt_1,164,196). coords(br1gt_1,34,156,196).  visible(br1gt_1,196). happensAt(walking(br1gt_1),196). orientation(br1gt_2,171,196). coords(br1gt_2,86,207,196).  visible(br1gt_2,196). happensAt(inactive(br1gt_2),196).
orientation(br1gt_1,164,197). coords(br1gt_1,33,155,197).  visible(br1gt_1,197). happensAt(walking(br1gt_1),197). orientation(br1gt_2,171,197). coords(br1gt_2,86,207,197).  visible(br1gt_2,197). happensAt(inactive(br1gt_2),197).
orientation(br1gt_1,164,198). coords(br1gt_1,33,155,198).  visible(br1gt_1,198). happensAt(walking(br1gt_1),198). orientation(br1gt_2,171,198). coords(br1gt_2,86,207,198).  visible(br1gt_2,198). happensAt(inactive(br1gt_2),198).
orientation(br1gt_1,164,199). coords(br1gt_1,32,155,199).  visible(br1gt_1,199). happensAt(walking(br1gt_1),199). orientation(br1gt_2,171,199). coords(br1gt_2,86,207,199).  visible(br1gt_2,199). happensAt(inactive(br1gt_2),199).
orientation(br1gt_1,164,200). coords(br1gt_1,33,155,200).  visible(br1gt_1,200). happensAt(walking(br1gt_1),200). orientation(br1gt_2,171,200). coords(br1gt_2,86,207,200).  visible(br1gt_2,200). happensAt(inactive(br1gt_2),200).
orientation(br1gt_1,164,201). coords(br1gt_1,32,157,201).  visible(br1gt_1,201). happensAt(walking(br1gt_1),201). orientation(br1gt_2,171,201). coords(br1gt_2,86,207,201).  visible(br1gt_2,201). happensAt(inactive(br1gt_2),201).
orientation(br1gt_1,164,202). coords(br1gt_1,32,157,202).  visible(br1gt_1,202). happensAt(walking(br1gt_1),202). orientation(br1gt_2,171,202). coords(br1gt_2,86,207,202).  visible(br1gt_2,202). happensAt(inactive(br1gt_2),202).
orientation(br1gt_1,164,203). coords(br1gt_1,32,156,203).  visible(br1gt_1,203). happensAt(walking(br1gt_1),203). orientation(br1gt_2,171,203). coords(br1gt_2,86,207,203).  visible(br1gt_2,203). happensAt(inactive(br1gt_2),203).
orientation(br1gt_1,164,204). coords(br1gt_1,32,157,204).  visible(br1gt_1,204). happensAt(walking(br1gt_1),204). orientation(br1gt_2,171,204). coords(br1gt_2,86,207,204).  visible(br1gt_2,204). happensAt(inactive(br1gt_2),204).
#show holdsAt/2.