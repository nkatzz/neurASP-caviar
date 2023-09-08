
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

	orientation(wk2gt_2,141,178). coords(wk2gt_2,98,62,178).  visible(wk2gt_2,178). happensAt(inactive(wk2gt_2),178). orientation(wk2gt_3,164,178). coords(wk2gt_3,210,183,178).  visible(wk2gt_3,178). happensAt(walking(wk2gt_3),178).
orientation(wk2gt_2,141,179). coords(wk2gt_2,98,62,179).  visible(wk2gt_2,179). happensAt(inactive(wk2gt_2),179). orientation(wk2gt_3,164,179). coords(wk2gt_3,213,185,179).  visible(wk2gt_3,179). happensAt(walking(wk2gt_3),179).
orientation(wk2gt_2,141,180). coords(wk2gt_2,98,62,180).  visible(wk2gt_2,180). happensAt(inactive(wk2gt_2),180). orientation(wk2gt_3,169,180). coords(wk2gt_3,217,187,180).  visible(wk2gt_3,180). happensAt(walking(wk2gt_3),180).
orientation(wk2gt_2,141,181). coords(wk2gt_2,98,62,181).  visible(wk2gt_2,181). happensAt(inactive(wk2gt_2),181). orientation(wk2gt_3,169,181). coords(wk2gt_3,220,189,181).  visible(wk2gt_3,181). happensAt(walking(wk2gt_3),181).
orientation(wk2gt_2,141,182). coords(wk2gt_2,98,62,182).  visible(wk2gt_2,182). happensAt(inactive(wk2gt_2),182). orientation(wk2gt_3,169,182). coords(wk2gt_3,223,191,182).  visible(wk2gt_3,182). happensAt(walking(wk2gt_3),182).
orientation(wk2gt_2,141,183). coords(wk2gt_2,98,62,183).  visible(wk2gt_2,183). happensAt(inactive(wk2gt_2),183). orientation(wk2gt_3,169,183). coords(wk2gt_3,225,194,183).  visible(wk2gt_3,183). happensAt(walking(wk2gt_3),183).
orientation(wk2gt_2,141,184). coords(wk2gt_2,98,62,184).  visible(wk2gt_2,184). happensAt(inactive(wk2gt_2),184). orientation(wk2gt_3,169,184). coords(wk2gt_3,227,197,184).  visible(wk2gt_3,184). happensAt(walking(wk2gt_3),184).
orientation(wk2gt_2,141,185). coords(wk2gt_2,98,62,185).  visible(wk2gt_2,185). happensAt(inactive(wk2gt_2),185). orientation(wk2gt_3,165,185). coords(wk2gt_3,230,200,185).  visible(wk2gt_3,185). happensAt(walking(wk2gt_3),185).
orientation(wk2gt_2,141,186). coords(wk2gt_2,98,62,186).  visible(wk2gt_2,186). happensAt(inactive(wk2gt_2),186). orientation(wk2gt_3,165,186). coords(wk2gt_3,232,202,186).  visible(wk2gt_3,186). happensAt(walking(wk2gt_3),186).
orientation(wk2gt_2,141,187). coords(wk2gt_2,98,62,187).  visible(wk2gt_2,187). happensAt(inactive(wk2gt_2),187). orientation(wk2gt_3,165,187). coords(wk2gt_3,233,205,187).  visible(wk2gt_3,187). happensAt(walking(wk2gt_3),187).
orientation(wk2gt_2,141,188). coords(wk2gt_2,98,62,188).  visible(wk2gt_2,188). happensAt(inactive(wk2gt_2),188). orientation(wk2gt_3,157,188). coords(wk2gt_3,234,208,188).  visible(wk2gt_3,188). happensAt(walking(wk2gt_3),188).
orientation(wk2gt_2,141,189). coords(wk2gt_2,98,62,189).  visible(wk2gt_2,189). happensAt(inactive(wk2gt_2),189). orientation(wk2gt_3,157,189). coords(wk2gt_3,235,211,189).  visible(wk2gt_3,189). happensAt(walking(wk2gt_3),189).
orientation(wk2gt_2,141,190). coords(wk2gt_2,98,62,190).  visible(wk2gt_2,190). happensAt(inactive(wk2gt_2),190). orientation(wk2gt_3,157,190). coords(wk2gt_3,237,216,190).  visible(wk2gt_3,190). happensAt(walking(wk2gt_3),190).
orientation(wk2gt_2,141,191). coords(wk2gt_2,98,62,191).  visible(wk2gt_2,191). happensAt(inactive(wk2gt_2),191). orientation(wk2gt_3,162,191). coords(wk2gt_3,239,220,191).  visible(wk2gt_3,191). happensAt(walking(wk2gt_3),191).
orientation(wk2gt_2,141,192). coords(wk2gt_2,98,62,192).  visible(wk2gt_2,192). happensAt(inactive(wk2gt_2),192). orientation(wk2gt_3,169,192). coords(wk2gt_3,242,224,192).  visible(wk2gt_3,192). happensAt(walking(wk2gt_3),192).
orientation(wk2gt_2,141,193). coords(wk2gt_2,98,62,193).  visible(wk2gt_2,193). happensAt(inactive(wk2gt_2),193). orientation(wk2gt_3,174,193). coords(wk2gt_3,245,226,193).  visible(wk2gt_3,193). happensAt(walking(wk2gt_3),193).
orientation(wk2gt_2,141,194). coords(wk2gt_2,98,62,194).  visible(wk2gt_2,194). happensAt(inactive(wk2gt_2),194). orientation(wk2gt_3,170,194). coords(wk2gt_3,248,228,194).  visible(wk2gt_3,194). happensAt(walking(wk2gt_3),194).
orientation(wk2gt_2,141,195). coords(wk2gt_2,98,62,195).  visible(wk2gt_2,195). happensAt(inactive(wk2gt_2),195). orientation(wk2gt_3,9,195). coords(wk2gt_3,250,228,195).  visible(wk2gt_3,195). happensAt(walking(wk2gt_3),195).
orientation(wk2gt_2,141,196). coords(wk2gt_2,98,62,196).  visible(wk2gt_2,196). happensAt(inactive(wk2gt_2),196). orientation(wk2gt_3,29,196). coords(wk2gt_3,250,230,196).  visible(wk2gt_3,196). happensAt(walking(wk2gt_3),196).
orientation(wk2gt_2,141,197). coords(wk2gt_2,98,62,197).  visible(wk2gt_2,197). happensAt(inactive(wk2gt_2),197). orientation(wk2gt_3,40,197). coords(wk2gt_3,251,231,197).  visible(wk2gt_3,197). happensAt(walking(wk2gt_3),197).
orientation(wk2gt_2,141,198). coords(wk2gt_2,98,62,198).  visible(wk2gt_2,198). happensAt(inactive(wk2gt_2),198). orientation(wk2gt_3,37,198). coords(wk2gt_3,253,233,198).  visible(wk2gt_3,198). happensAt(walking(wk2gt_3),198).
orientation(wk2gt_2,141,199). coords(wk2gt_2,98,62,199).  visible(wk2gt_2,199). happensAt(inactive(wk2gt_2),199). orientation(wk2gt_3,38,199). coords(wk2gt_3,255,237,199).  visible(wk2gt_3,199). happensAt(walking(wk2gt_3),199).
orientation(wk2gt_2,141,200). coords(wk2gt_2,98,62,200).  visible(wk2gt_2,200). happensAt(inactive(wk2gt_2),200). orientation(wk2gt_3,38,200). coords(wk2gt_3,258,241,200).  visible(wk2gt_3,200). happensAt(walking(wk2gt_3),200).
orientation(wk2gt_2,141,201). coords(wk2gt_2,98,62,201).  visible(wk2gt_2,201). happensAt(inactive(wk2gt_2),201). orientation(wk2gt_3,38,201). coords(wk2gt_3,262,245,201).  visible(wk2gt_3,201). happensAt(walking(wk2gt_3),201).
orientation(wk2gt_2,141,202). coords(wk2gt_2,98,62,202).  visible(wk2gt_2,202). happensAt(inactive(wk2gt_2),202). orientation(wk2gt_3,38,202). coords(wk2gt_3,265,247,202).  visible(wk2gt_3,202). happensAt(walking(wk2gt_3),202).
#show holdsAt/2.