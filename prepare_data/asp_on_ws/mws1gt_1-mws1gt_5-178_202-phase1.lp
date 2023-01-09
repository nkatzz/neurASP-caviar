
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

	orientation(mws1gt_1,155,178). coords(mws1gt_1,151,136,178).  visible(mws1gt_1,178). happensAt(walking(mws1gt_1),178). orientation(mws1gt_5,145,178). coords(mws1gt_5,59,93,178).  visible(mws1gt_5,178). happensAt(walking(mws1gt_5),178).
orientation(mws1gt_1,155,179). coords(mws1gt_1,151,136,179).  visible(mws1gt_1,179). happensAt(walking(mws1gt_1),179). orientation(mws1gt_5,145,179). coords(mws1gt_5,59,94,179).  visible(mws1gt_5,179). happensAt(walking(mws1gt_5),179).
orientation(mws1gt_1,155,180). coords(mws1gt_1,151,136,180).  visible(mws1gt_1,180). happensAt(walking(mws1gt_1),180). orientation(mws1gt_5,145,180). coords(mws1gt_5,58,95,180).  visible(mws1gt_5,180). happensAt(walking(mws1gt_5),180).
orientation(mws1gt_1,155,181). coords(mws1gt_1,151,136,181).  visible(mws1gt_1,181). happensAt(walking(mws1gt_1),181). orientation(mws1gt_5,145,181). coords(mws1gt_5,58,96,181).  visible(mws1gt_5,181). happensAt(walking(mws1gt_5),181).
orientation(mws1gt_1,155,182). coords(mws1gt_1,150,135,182).  visible(mws1gt_1,182). happensAt(walking(mws1gt_1),182). orientation(mws1gt_5,145,182). coords(mws1gt_5,58,96,182).  visible(mws1gt_5,182). happensAt(walking(mws1gt_5),182).
orientation(mws1gt_1,155,183). coords(mws1gt_1,150,135,183).  visible(mws1gt_1,183). happensAt(walking(mws1gt_1),183). orientation(mws1gt_5,145,183). coords(mws1gt_5,58,97,183).  visible(mws1gt_5,183). happensAt(walking(mws1gt_5),183).
orientation(mws1gt_1,155,184). coords(mws1gt_1,149,136,184).  visible(mws1gt_1,184). happensAt(walking(mws1gt_1),184). orientation(mws1gt_5,145,184). coords(mws1gt_5,58,97,184).  visible(mws1gt_5,184). happensAt(walking(mws1gt_5),184).
orientation(mws1gt_1,155,185). coords(mws1gt_1,149,136,185).  visible(mws1gt_1,185). happensAt(walking(mws1gt_1),185). orientation(mws1gt_5,145,185). coords(mws1gt_5,57,97,185).  visible(mws1gt_5,185). happensAt(walking(mws1gt_5),185).
orientation(mws1gt_1,155,186). coords(mws1gt_1,148,136,186).  visible(mws1gt_1,186). happensAt(walking(mws1gt_1),186). orientation(mws1gt_5,145,186). coords(mws1gt_5,57,97,186).  visible(mws1gt_5,186). happensAt(walking(mws1gt_5),186).
orientation(mws1gt_1,155,187). coords(mws1gt_1,148,136,187).  visible(mws1gt_1,187). happensAt(walking(mws1gt_1),187). orientation(mws1gt_5,145,187). coords(mws1gt_5,57,98,187).  visible(mws1gt_5,187). happensAt(walking(mws1gt_5),187).
orientation(mws1gt_1,155,188). coords(mws1gt_1,147,136,188).  visible(mws1gt_1,188). happensAt(walking(mws1gt_1),188). orientation(mws1gt_5,145,188). coords(mws1gt_5,56,98,188).  visible(mws1gt_5,188). happensAt(walking(mws1gt_5),188).
orientation(mws1gt_1,155,189). coords(mws1gt_1,146,136,189).  visible(mws1gt_1,189). happensAt(walking(mws1gt_1),189). orientation(mws1gt_5,145,189). coords(mws1gt_5,55,99,189).  visible(mws1gt_5,189). happensAt(walking(mws1gt_5),189).
orientation(mws1gt_1,155,190). coords(mws1gt_1,145,137,190).  visible(mws1gt_1,190). happensAt(walking(mws1gt_1),190). orientation(mws1gt_5,145,190). coords(mws1gt_5,55,99,190).  visible(mws1gt_5,190). happensAt(walking(mws1gt_5),190).
orientation(mws1gt_1,155,191). coords(mws1gt_1,144,137,191).  visible(mws1gt_1,191). happensAt(walking(mws1gt_1),191). orientation(mws1gt_5,145,191). coords(mws1gt_5,54,99,191).  visible(mws1gt_5,191). happensAt(walking(mws1gt_5),191).
orientation(mws1gt_1,155,192). coords(mws1gt_1,144,136,192).  visible(mws1gt_1,192). happensAt(walking(mws1gt_1),192). orientation(mws1gt_5,145,192). coords(mws1gt_5,53,99,192).  visible(mws1gt_5,192). happensAt(walking(mws1gt_5),192).
orientation(mws1gt_1,155,193). coords(mws1gt_1,144,136,193).  visible(mws1gt_1,193). happensAt(walking(mws1gt_1),193). orientation(mws1gt_5,145,193). coords(mws1gt_5,53,99,193).  visible(mws1gt_5,193). happensAt(walking(mws1gt_5),193).
orientation(mws1gt_1,155,194). coords(mws1gt_1,143,136,194).  visible(mws1gt_1,194). happensAt(walking(mws1gt_1),194). orientation(mws1gt_5,145,194). coords(mws1gt_5,49,99,194).  visible(mws1gt_5,194). happensAt(walking(mws1gt_5),194).
orientation(mws1gt_1,155,195). coords(mws1gt_1,143,135,195).  visible(mws1gt_1,195). happensAt(walking(mws1gt_1),195). orientation(mws1gt_5,145,195). coords(mws1gt_5,49,99,195).  visible(mws1gt_5,195). happensAt(walking(mws1gt_5),195).
orientation(mws1gt_1,155,196). coords(mws1gt_1,143,135,196).  visible(mws1gt_1,196). happensAt(walking(mws1gt_1),196). orientation(mws1gt_5,145,196). coords(mws1gt_5,49,99,196).  visible(mws1gt_5,196). happensAt(walking(mws1gt_5),196).
orientation(mws1gt_1,155,197). coords(mws1gt_1,142,135,197).  visible(mws1gt_1,197). happensAt(walking(mws1gt_1),197). orientation(mws1gt_5,158,197). coords(mws1gt_5,48,100,197).  visible(mws1gt_5,197). happensAt(walking(mws1gt_5),197).
orientation(mws1gt_1,155,198). coords(mws1gt_1,142,134,198).  visible(mws1gt_1,198). happensAt(walking(mws1gt_1),198). orientation(mws1gt_5,158,198). coords(mws1gt_5,48,100,198).  visible(mws1gt_5,198). happensAt(walking(mws1gt_5),198).
orientation(mws1gt_1,155,199). coords(mws1gt_1,142,134,199).  visible(mws1gt_1,199). happensAt(walking(mws1gt_1),199). orientation(mws1gt_5,161,199). coords(mws1gt_5,48,102,199).  visible(mws1gt_5,199). happensAt(walking(mws1gt_5),199).
orientation(mws1gt_1,155,200). coords(mws1gt_1,140,133,200).  visible(mws1gt_1,200). happensAt(walking(mws1gt_1),200). orientation(mws1gt_5,161,200). coords(mws1gt_5,48,102,200).  visible(mws1gt_5,200). happensAt(walking(mws1gt_5),200).
orientation(mws1gt_1,155,201). coords(mws1gt_1,139,133,201).  visible(mws1gt_1,201). happensAt(walking(mws1gt_1),201). orientation(mws1gt_5,161,201). coords(mws1gt_5,48,102,201).  visible(mws1gt_5,201). happensAt(walking(mws1gt_5),201).
orientation(mws1gt_1,155,202). coords(mws1gt_1,139,133,202).  visible(mws1gt_1,202). happensAt(walking(mws1gt_1),202). orientation(mws1gt_5,161,202). coords(mws1gt_5,47,103,202).  visible(mws1gt_5,202). happensAt(walking(mws1gt_5),202).
#show holdsAt/2.