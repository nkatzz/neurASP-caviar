
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

	orientation(mws1gt_1,155,180). coords(mws1gt_1,151,136,180).  visible(mws1gt_1,180). happensAt(walking(mws1gt_1),180). orientation(mws1gt_6,140,180). coords(mws1gt_6,70,73,180).  visible(mws1gt_6,180). happensAt(walking(mws1gt_6),180).
orientation(mws1gt_1,155,181). coords(mws1gt_1,151,136,181).  visible(mws1gt_1,181). happensAt(walking(mws1gt_1),181). orientation(mws1gt_6,140,181). coords(mws1gt_6,69,74,181).  visible(mws1gt_6,181). happensAt(walking(mws1gt_6),181).
orientation(mws1gt_1,155,182). coords(mws1gt_1,150,135,182).  visible(mws1gt_1,182). happensAt(walking(mws1gt_1),182). orientation(mws1gt_6,140,182). coords(mws1gt_6,69,74,182).  visible(mws1gt_6,182). happensAt(walking(mws1gt_6),182).
orientation(mws1gt_1,155,183). coords(mws1gt_1,150,135,183).  visible(mws1gt_1,183). happensAt(walking(mws1gt_1),183). orientation(mws1gt_6,140,183). coords(mws1gt_6,69,75,183).  visible(mws1gt_6,183). happensAt(walking(mws1gt_6),183).
orientation(mws1gt_1,155,184). coords(mws1gt_1,149,136,184).  visible(mws1gt_1,184). happensAt(walking(mws1gt_1),184). orientation(mws1gt_6,140,184). coords(mws1gt_6,68,76,184).  visible(mws1gt_6,184). happensAt(walking(mws1gt_6),184).
orientation(mws1gt_1,155,185). coords(mws1gt_1,149,136,185).  visible(mws1gt_1,185). happensAt(walking(mws1gt_1),185). orientation(mws1gt_6,140,185). coords(mws1gt_6,68,76,185).  visible(mws1gt_6,185). happensAt(walking(mws1gt_6),185).
orientation(mws1gt_1,155,186). coords(mws1gt_1,148,136,186).  visible(mws1gt_1,186). happensAt(walking(mws1gt_1),186). orientation(mws1gt_6,140,186). coords(mws1gt_6,68,76,186).  visible(mws1gt_6,186). happensAt(walking(mws1gt_6),186).
orientation(mws1gt_1,155,187). coords(mws1gt_1,148,136,187).  visible(mws1gt_1,187). happensAt(walking(mws1gt_1),187). orientation(mws1gt_6,140,187). coords(mws1gt_6,68,77,187).  visible(mws1gt_6,187). happensAt(walking(mws1gt_6),187).
orientation(mws1gt_1,155,188). coords(mws1gt_1,147,136,188).  visible(mws1gt_1,188). happensAt(walking(mws1gt_1),188). orientation(mws1gt_6,140,188). coords(mws1gt_6,67,77,188).  visible(mws1gt_6,188). happensAt(walking(mws1gt_6),188).
orientation(mws1gt_1,155,189). coords(mws1gt_1,146,136,189).  visible(mws1gt_1,189). happensAt(walking(mws1gt_1),189). orientation(mws1gt_6,140,189). coords(mws1gt_6,66,78,189).  visible(mws1gt_6,189). happensAt(walking(mws1gt_6),189).
orientation(mws1gt_1,155,190). coords(mws1gt_1,145,137,190).  visible(mws1gt_1,190). happensAt(walking(mws1gt_1),190). orientation(mws1gt_6,140,190). coords(mws1gt_6,66,79,190).  visible(mws1gt_6,190). happensAt(walking(mws1gt_6),190).
orientation(mws1gt_1,155,191). coords(mws1gt_1,144,137,191).  visible(mws1gt_1,191). happensAt(walking(mws1gt_1),191). orientation(mws1gt_6,140,191). coords(mws1gt_6,66,80,191).  visible(mws1gt_6,191). happensAt(walking(mws1gt_6),191).
orientation(mws1gt_1,155,192). coords(mws1gt_1,144,136,192).  visible(mws1gt_1,192). happensAt(walking(mws1gt_1),192). orientation(mws1gt_6,140,192). coords(mws1gt_6,66,81,192).  visible(mws1gt_6,192). happensAt(walking(mws1gt_6),192).
orientation(mws1gt_1,155,193). coords(mws1gt_1,144,136,193).  visible(mws1gt_1,193). happensAt(walking(mws1gt_1),193). orientation(mws1gt_6,140,193). coords(mws1gt_6,66,82,193).  visible(mws1gt_6,193). happensAt(walking(mws1gt_6),193).
orientation(mws1gt_1,155,194). coords(mws1gt_1,143,136,194).  visible(mws1gt_1,194). happensAt(walking(mws1gt_1),194). orientation(mws1gt_6,140,194). coords(mws1gt_6,66,83,194).  visible(mws1gt_6,194). happensAt(walking(mws1gt_6),194).
orientation(mws1gt_1,155,195). coords(mws1gt_1,143,135,195).  visible(mws1gt_1,195). happensAt(walking(mws1gt_1),195). orientation(mws1gt_6,140,195). coords(mws1gt_6,66,84,195).  visible(mws1gt_6,195). happensAt(walking(mws1gt_6),195).
orientation(mws1gt_1,155,196). coords(mws1gt_1,143,135,196).  visible(mws1gt_1,196). happensAt(walking(mws1gt_1),196). orientation(mws1gt_6,140,196). coords(mws1gt_6,65,84,196).  visible(mws1gt_6,196). happensAt(walking(mws1gt_6),196).
orientation(mws1gt_1,155,197). coords(mws1gt_1,142,135,197).  visible(mws1gt_1,197). happensAt(walking(mws1gt_1),197). orientation(mws1gt_6,140,197). coords(mws1gt_6,64,84,197).  visible(mws1gt_6,197). happensAt(walking(mws1gt_6),197).
orientation(mws1gt_1,155,198). coords(mws1gt_1,142,134,198).  visible(mws1gt_1,198). happensAt(walking(mws1gt_1),198). orientation(mws1gt_6,140,198). coords(mws1gt_6,64,84,198).  visible(mws1gt_6,198). happensAt(walking(mws1gt_6),198).
orientation(mws1gt_1,155,199). coords(mws1gt_1,142,134,199).  visible(mws1gt_1,199). happensAt(walking(mws1gt_1),199). orientation(mws1gt_6,140,199). coords(mws1gt_6,64,85,199).  visible(mws1gt_6,199). happensAt(walking(mws1gt_6),199).
orientation(mws1gt_1,155,200). coords(mws1gt_1,140,133,200).  visible(mws1gt_1,200). happensAt(walking(mws1gt_1),200). orientation(mws1gt_6,140,200). coords(mws1gt_6,63,85,200).  visible(mws1gt_6,200). happensAt(walking(mws1gt_6),200).
orientation(mws1gt_1,155,201). coords(mws1gt_1,139,133,201).  visible(mws1gt_1,201). happensAt(walking(mws1gt_1),201). orientation(mws1gt_6,140,201). coords(mws1gt_6,62,86,201).  visible(mws1gt_6,201). happensAt(walking(mws1gt_6),201).
orientation(mws1gt_1,155,202). coords(mws1gt_1,139,133,202).  visible(mws1gt_1,202). happensAt(walking(mws1gt_1),202). orientation(mws1gt_6,140,202). coords(mws1gt_6,61,86,202).  visible(mws1gt_6,202). happensAt(walking(mws1gt_6),202).
orientation(mws1gt_1,155,203). coords(mws1gt_1,138,133,203).  visible(mws1gt_1,203). happensAt(walking(mws1gt_1),203). orientation(mws1gt_6,140,203). coords(mws1gt_6,60,87,203).  visible(mws1gt_6,203). happensAt(walking(mws1gt_6),203).
orientation(mws1gt_1,155,204). coords(mws1gt_1,138,133,204).  visible(mws1gt_1,204). happensAt(walking(mws1gt_1),204). orientation(mws1gt_6,140,204). coords(mws1gt_6,60,88,204).  visible(mws1gt_6,204). happensAt(walking(mws1gt_6),204).
#show holdsAt/2.