
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

	orientation(mws1gt_4,0,180). coords(mws1gt_4,30,118,180).  visible(mws1gt_4,180). happensAt(walking(mws1gt_4),180). orientation(mws1gt_6,140,180). coords(mws1gt_6,70,73,180).  visible(mws1gt_6,180). happensAt(walking(mws1gt_6),180).
orientation(mws1gt_4,0,181). coords(mws1gt_4,30,118,181).  visible(mws1gt_4,181). happensAt(walking(mws1gt_4),181). orientation(mws1gt_6,140,181). coords(mws1gt_6,69,74,181).  visible(mws1gt_6,181). happensAt(walking(mws1gt_6),181).
orientation(mws1gt_4,0,182). coords(mws1gt_4,30,118,182).  visible(mws1gt_4,182). happensAt(walking(mws1gt_4),182). orientation(mws1gt_6,140,182). coords(mws1gt_6,69,74,182).  visible(mws1gt_6,182). happensAt(walking(mws1gt_6),182).
orientation(mws1gt_4,0,183). coords(mws1gt_4,30,117,183).  visible(mws1gt_4,183). happensAt(walking(mws1gt_4),183). orientation(mws1gt_6,140,183). coords(mws1gt_6,69,75,183).  visible(mws1gt_6,183). happensAt(walking(mws1gt_6),183).
orientation(mws1gt_4,0,184). coords(mws1gt_4,30,117,184).  visible(mws1gt_4,184). happensAt(walking(mws1gt_4),184). orientation(mws1gt_6,140,184). coords(mws1gt_6,68,76,184).  visible(mws1gt_6,184). happensAt(walking(mws1gt_6),184).
orientation(mws1gt_4,0,185). coords(mws1gt_4,30,117,185).  visible(mws1gt_4,185). happensAt(walking(mws1gt_4),185). orientation(mws1gt_6,140,185). coords(mws1gt_6,68,76,185).  visible(mws1gt_6,185). happensAt(walking(mws1gt_6),185).
orientation(mws1gt_4,0,186). coords(mws1gt_4,30,116,186).  visible(mws1gt_4,186). happensAt(walking(mws1gt_4),186). orientation(mws1gt_6,140,186). coords(mws1gt_6,68,76,186).  visible(mws1gt_6,186). happensAt(walking(mws1gt_6),186).
orientation(mws1gt_4,0,187). coords(mws1gt_4,30,115,187).  visible(mws1gt_4,187). happensAt(walking(mws1gt_4),187). orientation(mws1gt_6,140,187). coords(mws1gt_6,68,77,187).  visible(mws1gt_6,187). happensAt(walking(mws1gt_6),187).
orientation(mws1gt_4,0,188). coords(mws1gt_4,30,115,188).  visible(mws1gt_4,188). happensAt(walking(mws1gt_4),188). orientation(mws1gt_6,140,188). coords(mws1gt_6,67,77,188).  visible(mws1gt_6,188). happensAt(walking(mws1gt_6),188).
orientation(mws1gt_4,0,189). coords(mws1gt_4,31,114,189).  visible(mws1gt_4,189). happensAt(walking(mws1gt_4),189). orientation(mws1gt_6,140,189). coords(mws1gt_6,66,78,189).  visible(mws1gt_6,189). happensAt(walking(mws1gt_6),189).
orientation(mws1gt_4,0,190). coords(mws1gt_4,31,114,190).  visible(mws1gt_4,190). happensAt(walking(mws1gt_4),190). orientation(mws1gt_6,140,190). coords(mws1gt_6,66,79,190).  visible(mws1gt_6,190). happensAt(walking(mws1gt_6),190).
orientation(mws1gt_4,0,191). coords(mws1gt_4,31,114,191).  visible(mws1gt_4,191). happensAt(walking(mws1gt_4),191). orientation(mws1gt_6,140,191). coords(mws1gt_6,66,80,191).  visible(mws1gt_6,191). happensAt(walking(mws1gt_6),191).
orientation(mws1gt_4,0,192). coords(mws1gt_4,31,114,192).  visible(mws1gt_4,192). happensAt(walking(mws1gt_4),192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192). happensAt(walking(mws1gt_6),192).
orientation(mws1gt_4,0,193). coords(mws1gt_4,31,113,193).  visible(mws1gt_4,193). happensAt(walking(mws1gt_4),193). orientation(mws1gt_6,140,193). coords(mws1gt_6,66,82,193).  visible(mws1gt_6,193). happensAt(walking(mws1gt_6),193).
orientation(mws1gt_4,0,194). coords(mws1gt_4,31,113,194).  visible(mws1gt_4,194). happensAt(walking(mws1gt_4),194). orientation(mws1gt_6,140,194). coords(mws1gt_6,66,83,194).  visible(mws1gt_6,194). happensAt(walking(mws1gt_6),194).
orientation(mws1gt_4,0,195). coords(mws1gt_4,31,113,195).  visible(mws1gt_4,195). happensAt(walking(mws1gt_4),195). orientation(mws1gt_6,140,195). coords(mws1gt_6,66,84,195).  visible(mws1gt_6,195). happensAt(walking(mws1gt_6),195).
orientation(mws1gt_4,0,196). coords(mws1gt_4,32,113,196).  visible(mws1gt_4,196). happensAt(walking(mws1gt_4),196). orientation(mws1gt_6,140,196). coords(mws1gt_6,65,84,196).  visible(mws1gt_6,196). happensAt(walking(mws1gt_6),196).
orientation(mws1gt_4,0,197). coords(mws1gt_4,32,112,197).  visible(mws1gt_4,197). happensAt(walking(mws1gt_4),197). orientation(mws1gt_6,140,197). coords(mws1gt_6,64,84,197).  visible(mws1gt_6,197). happensAt(walking(mws1gt_6),197).
orientation(mws1gt_4,0,198). coords(mws1gt_4,32,112,198).  visible(mws1gt_4,198). happensAt(walking(mws1gt_4),198). orientation(mws1gt_6,140,198). coords(mws1gt_6,64,84,198).  visible(mws1gt_6,198). happensAt(walking(mws1gt_6),198).
orientation(mws1gt_4,0,199). coords(mws1gt_4,32,111,199).  visible(mws1gt_4,199). happensAt(walking(mws1gt_4),199). orientation(mws1gt_6,140,199). coords(mws1gt_6,64,85,199).  visible(mws1gt_6,199). happensAt(walking(mws1gt_6),199).
orientation(mws1gt_4,0,200). coords(mws1gt_4,32,111,200).  visible(mws1gt_4,200). happensAt(walking(mws1gt_4),200). orientation(mws1gt_6,140,200). coords(mws1gt_6,63,85,200).  visible(mws1gt_6,200). happensAt(walking(mws1gt_6),200).
orientation(mws1gt_4,0,201). coords(mws1gt_4,32,111,201).  visible(mws1gt_4,201). happensAt(walking(mws1gt_4),201). orientation(mws1gt_6,140,201). coords(mws1gt_6,62,86,201).  visible(mws1gt_6,201). happensAt(walking(mws1gt_6),201).
orientation(mws1gt_4,0,202). coords(mws1gt_4,32,110,202).  visible(mws1gt_4,202). happensAt(walking(mws1gt_4),202). orientation(mws1gt_6,140,202). coords(mws1gt_6,61,86,202).  visible(mws1gt_6,202). happensAt(walking(mws1gt_6),202).
orientation(mws1gt_4,0,203). coords(mws1gt_4,32,110,203).  visible(mws1gt_4,203). happensAt(walking(mws1gt_4),203). orientation(mws1gt_6,140,203). coords(mws1gt_6,60,87,203).  visible(mws1gt_6,203). happensAt(walking(mws1gt_6),203).
orientation(mws1gt_4,0,204). coords(mws1gt_4,32,110,204).  visible(mws1gt_4,204). happensAt(walking(mws1gt_4),204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204). happensAt(walking(mws1gt_6),204).
#show holdsAt/2.