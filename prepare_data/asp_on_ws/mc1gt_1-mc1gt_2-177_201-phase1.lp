
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

	orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177). happensAt(walking(mc1gt_1),177). orientation(mc1gt_2,144,177). coords(mc1gt_2,88,113,177).  visible(mc1gt_2,177). happensAt(walking(mc1gt_2),177).
orientation(mc1gt_1,145,178). coords(mc1gt_1,139,91,178).  visible(mc1gt_1,178). happensAt(walking(mc1gt_1),178). orientation(mc1gt_2,144,178). coords(mc1gt_2,89,114,178).  visible(mc1gt_2,178). happensAt(walking(mc1gt_2),178).
orientation(mc1gt_1,145,179). coords(mc1gt_1,140,91,179).  visible(mc1gt_1,179). happensAt(walking(mc1gt_1),179). orientation(mc1gt_2,144,179). coords(mc1gt_2,90,114,179).  visible(mc1gt_2,179). happensAt(walking(mc1gt_2),179).
orientation(mc1gt_1,145,180). coords(mc1gt_1,140,91,180).  visible(mc1gt_1,180). happensAt(walking(mc1gt_1),180). orientation(mc1gt_2,144,180). coords(mc1gt_2,91,114,180).  visible(mc1gt_2,180). happensAt(walking(mc1gt_2),180).
orientation(mc1gt_1,145,181). coords(mc1gt_1,140,92,181).  visible(mc1gt_1,181). happensAt(walking(mc1gt_1),181). orientation(mc1gt_2,144,181). coords(mc1gt_2,92,114,181).  visible(mc1gt_2,181). happensAt(walking(mc1gt_2),181).
orientation(mc1gt_1,145,182). coords(mc1gt_1,142,92,182).  visible(mc1gt_1,182). happensAt(walking(mc1gt_1),182). orientation(mc1gt_2,144,182). coords(mc1gt_2,92,114,182).  visible(mc1gt_2,182). happensAt(walking(mc1gt_2),182).
orientation(mc1gt_1,145,183). coords(mc1gt_1,144,92,183).  visible(mc1gt_1,183). happensAt(walking(mc1gt_1),183). orientation(mc1gt_2,145,183). coords(mc1gt_2,92,115,183).  visible(mc1gt_2,183). happensAt(walking(mc1gt_2),183).
orientation(mc1gt_1,145,184). coords(mc1gt_1,146,92,184).  visible(mc1gt_1,184). happensAt(walking(mc1gt_1),184). orientation(mc1gt_2,146,184). coords(mc1gt_2,92,115,184).  visible(mc1gt_2,184). happensAt(walking(mc1gt_2),184).
orientation(mc1gt_1,145,185). coords(mc1gt_1,147,94,185).  visible(mc1gt_1,185). happensAt(walking(mc1gt_1),185). orientation(mc1gt_2,147,185). coords(mc1gt_2,92,115,185).  visible(mc1gt_2,185). happensAt(walking(mc1gt_2),185).
orientation(mc1gt_1,145,186). coords(mc1gt_1,148,94,186).  visible(mc1gt_1,186). happensAt(walking(mc1gt_1),186). orientation(mc1gt_2,147,186). coords(mc1gt_2,93,115,186).  visible(mc1gt_2,186). happensAt(walking(mc1gt_2),186).
orientation(mc1gt_1,145,187). coords(mc1gt_1,149,95,187).  visible(mc1gt_1,187). happensAt(walking(mc1gt_1),187). orientation(mc1gt_2,147,187). coords(mc1gt_2,93,115,187).  visible(mc1gt_2,187). happensAt(walking(mc1gt_2),187).
orientation(mc1gt_1,143,188). coords(mc1gt_1,150,96,188).  visible(mc1gt_1,188). happensAt(walking(mc1gt_1),188). orientation(mc1gt_2,147,188). coords(mc1gt_2,93,115,188).  visible(mc1gt_2,188). happensAt(walking(mc1gt_2),188).
orientation(mc1gt_1,141,189). coords(mc1gt_1,151,96,189).  visible(mc1gt_1,189). happensAt(walking(mc1gt_1),189). orientation(mc1gt_2,147,189). coords(mc1gt_2,95,115,189).  visible(mc1gt_2,189). happensAt(walking(mc1gt_2),189).
orientation(mc1gt_1,139,190). coords(mc1gt_1,151,97,190).  visible(mc1gt_1,190). happensAt(walking(mc1gt_1),190). orientation(mc1gt_2,148,190). coords(mc1gt_2,97,115,190).  visible(mc1gt_2,190). happensAt(walking(mc1gt_2),190).
orientation(mc1gt_1,139,191). coords(mc1gt_1,152,97,191).  visible(mc1gt_1,191). happensAt(walking(mc1gt_1),191). orientation(mc1gt_2,149,191). coords(mc1gt_2,98,116,191).  visible(mc1gt_2,191). happensAt(walking(mc1gt_2),191).
orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192). happensAt(walking(mc1gt_1),192). orientation(mc1gt_2,149,192). coords(mc1gt_2,99,116,192).  visible(mc1gt_2,192). happensAt(walking(mc1gt_2),192).
orientation(mc1gt_1,140,193). coords(mc1gt_1,152,97,193).  visible(mc1gt_1,193). happensAt(walking(mc1gt_1),193). orientation(mc1gt_2,149,193). coords(mc1gt_2,100,116,193).  visible(mc1gt_2,193). happensAt(walking(mc1gt_2),193).
orientation(mc1gt_1,140,194). coords(mc1gt_1,152,97,194).  visible(mc1gt_1,194). happensAt(walking(mc1gt_1),194). orientation(mc1gt_2,149,194). coords(mc1gt_2,101,117,194).  visible(mc1gt_2,194). happensAt(walking(mc1gt_2),194).
orientation(mc1gt_1,140,195). coords(mc1gt_1,152,97,195).  visible(mc1gt_1,195). happensAt(walking(mc1gt_1),195). orientation(mc1gt_2,148,195). coords(mc1gt_2,101,117,195).  visible(mc1gt_2,195). happensAt(walking(mc1gt_2),195).
orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196). happensAt(walking(mc1gt_1),196). orientation(mc1gt_2,148,196). coords(mc1gt_2,102,117,196).  visible(mc1gt_2,196). happensAt(walking(mc1gt_2),196).
orientation(mc1gt_1,140,197). coords(mc1gt_1,153,100,197).  visible(mc1gt_1,197). happensAt(walking(mc1gt_1),197). orientation(mc1gt_2,148,197). coords(mc1gt_2,102,117,197).  visible(mc1gt_2,197). happensAt(walking(mc1gt_2),197).
orientation(mc1gt_1,139,198). coords(mc1gt_1,154,101,198).  visible(mc1gt_1,198). happensAt(walking(mc1gt_1),198). orientation(mc1gt_2,147,198). coords(mc1gt_2,102,117,198).  visible(mc1gt_2,198). happensAt(walking(mc1gt_2),198).
orientation(mc1gt_1,139,199). coords(mc1gt_1,155,103,199).  visible(mc1gt_1,199). happensAt(walking(mc1gt_1),199). orientation(mc1gt_2,147,199). coords(mc1gt_2,102,117,199).  visible(mc1gt_2,199). happensAt(walking(mc1gt_2),199).
orientation(mc1gt_1,139,200). coords(mc1gt_1,157,105,200).  visible(mc1gt_1,200). happensAt(walking(mc1gt_1),200). orientation(mc1gt_2,147,200). coords(mc1gt_2,103,118,200).  visible(mc1gt_2,200). happensAt(walking(mc1gt_2),200).
orientation(mc1gt_1,137,201). coords(mc1gt_1,159,106,201).  visible(mc1gt_1,201). happensAt(walking(mc1gt_1),201). orientation(mc1gt_2,147,201). coords(mc1gt_2,103,118,201).  visible(mc1gt_2,201). happensAt(walking(mc1gt_2),201).
#show holdsAt/2.