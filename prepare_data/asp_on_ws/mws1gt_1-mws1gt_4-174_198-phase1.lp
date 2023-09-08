
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

	orientation(mws1gt_1,155,174). coords(mws1gt_1,150,139,174).  visible(mws1gt_1,174). happensAt(walking(mws1gt_1),174). orientation(mws1gt_4,0,174). coords(mws1gt_4,29,121,174).  visible(mws1gt_4,174). happensAt(walking(mws1gt_4),174).
orientation(mws1gt_1,155,175). coords(mws1gt_1,150,139,175).  visible(mws1gt_1,175). happensAt(walking(mws1gt_1),175). orientation(mws1gt_4,0,175). coords(mws1gt_4,29,120,175).  visible(mws1gt_4,175). happensAt(walking(mws1gt_4),175).
orientation(mws1gt_1,155,176). coords(mws1gt_1,150,138,176).  visible(mws1gt_1,176). happensAt(walking(mws1gt_1),176). orientation(mws1gt_4,0,176). coords(mws1gt_4,29,120,176).  visible(mws1gt_4,176). happensAt(walking(mws1gt_4),176).
orientation(mws1gt_1,155,177). coords(mws1gt_1,151,137,177).  visible(mws1gt_1,177). happensAt(walking(mws1gt_1),177). orientation(mws1gt_4,0,177). coords(mws1gt_4,29,120,177).  visible(mws1gt_4,177). happensAt(walking(mws1gt_4),177).
orientation(mws1gt_1,155,178). coords(mws1gt_1,151,136,178).  visible(mws1gt_1,178). happensAt(walking(mws1gt_1),178). orientation(mws1gt_4,0,178). coords(mws1gt_4,29,119,178).  visible(mws1gt_4,178). happensAt(walking(mws1gt_4),178).
orientation(mws1gt_1,155,179). coords(mws1gt_1,151,136,179).  visible(mws1gt_1,179). happensAt(walking(mws1gt_1),179). orientation(mws1gt_4,0,179). coords(mws1gt_4,29,119,179).  visible(mws1gt_4,179). happensAt(walking(mws1gt_4),179).
orientation(mws1gt_1,155,180). coords(mws1gt_1,151,136,180).  visible(mws1gt_1,180). happensAt(walking(mws1gt_1),180). orientation(mws1gt_4,0,180). coords(mws1gt_4,30,118,180).  visible(mws1gt_4,180). happensAt(walking(mws1gt_4),180).
orientation(mws1gt_1,155,181). coords(mws1gt_1,151,136,181).  visible(mws1gt_1,181). happensAt(walking(mws1gt_1),181). orientation(mws1gt_4,0,181). coords(mws1gt_4,30,118,181).  visible(mws1gt_4,181). happensAt(walking(mws1gt_4),181).
orientation(mws1gt_1,155,182). coords(mws1gt_1,150,135,182).  visible(mws1gt_1,182). happensAt(walking(mws1gt_1),182). orientation(mws1gt_4,0,182). coords(mws1gt_4,30,118,182).  visible(mws1gt_4,182). happensAt(walking(mws1gt_4),182).
orientation(mws1gt_1,155,183). coords(mws1gt_1,150,135,183).  visible(mws1gt_1,183). happensAt(walking(mws1gt_1),183). orientation(mws1gt_4,0,183). coords(mws1gt_4,30,117,183).  visible(mws1gt_4,183). happensAt(walking(mws1gt_4),183).
orientation(mws1gt_1,155,184). coords(mws1gt_1,149,136,184).  visible(mws1gt_1,184). happensAt(walking(mws1gt_1),184). orientation(mws1gt_4,0,184). coords(mws1gt_4,30,117,184).  visible(mws1gt_4,184). happensAt(walking(mws1gt_4),184).
orientation(mws1gt_1,155,185). coords(mws1gt_1,149,136,185).  visible(mws1gt_1,185). happensAt(walking(mws1gt_1),185). orientation(mws1gt_4,0,185). coords(mws1gt_4,30,117,185).  visible(mws1gt_4,185). happensAt(walking(mws1gt_4),185).
orientation(mws1gt_1,155,186). coords(mws1gt_1,148,136,186).  visible(mws1gt_1,186). happensAt(walking(mws1gt_1),186). orientation(mws1gt_4,0,186). coords(mws1gt_4,30,116,186).  visible(mws1gt_4,186). happensAt(walking(mws1gt_4),186).
orientation(mws1gt_1,155,187). coords(mws1gt_1,148,136,187).  visible(mws1gt_1,187). happensAt(walking(mws1gt_1),187). orientation(mws1gt_4,0,187). coords(mws1gt_4,30,115,187).  visible(mws1gt_4,187). happensAt(walking(mws1gt_4),187).
orientation(mws1gt_1,155,188). coords(mws1gt_1,147,136,188).  visible(mws1gt_1,188). happensAt(walking(mws1gt_1),188). orientation(mws1gt_4,0,188). coords(mws1gt_4,30,115,188).  visible(mws1gt_4,188). happensAt(walking(mws1gt_4),188).
orientation(mws1gt_1,155,189). coords(mws1gt_1,146,136,189).  visible(mws1gt_1,189). happensAt(walking(mws1gt_1),189). orientation(mws1gt_4,0,189). coords(mws1gt_4,31,114,189).  visible(mws1gt_4,189). happensAt(walking(mws1gt_4),189).
orientation(mws1gt_1,155,190). coords(mws1gt_1,145,137,190).  visible(mws1gt_1,190). happensAt(walking(mws1gt_1),190). orientation(mws1gt_4,0,190). coords(mws1gt_4,31,114,190).  visible(mws1gt_4,190). happensAt(walking(mws1gt_4),190).
orientation(mws1gt_1,155,191). coords(mws1gt_1,144,137,191).  visible(mws1gt_1,191). happensAt(walking(mws1gt_1),191). orientation(mws1gt_4,0,191). coords(mws1gt_4,31,114,191).  visible(mws1gt_4,191). happensAt(walking(mws1gt_4),191).
orientation(mws1gt_1,155,192). coords(mws1gt_1,144,136,192).  visible(mws1gt_1,192). happensAt(walking(mws1gt_1),192). orientation(mws1gt_4,0,192). coords(mws1gt_4,31,114,192).  visible(mws1gt_4,192). happensAt(walking(mws1gt_4),192).
orientation(mws1gt_1,155,193). coords(mws1gt_1,144,136,193).  visible(mws1gt_1,193). happensAt(walking(mws1gt_1),193). orientation(mws1gt_4,0,193). coords(mws1gt_4,31,113,193).  visible(mws1gt_4,193). happensAt(walking(mws1gt_4),193).
orientation(mws1gt_1,155,194). coords(mws1gt_1,143,136,194).  visible(mws1gt_1,194). happensAt(walking(mws1gt_1),194). orientation(mws1gt_4,0,194). coords(mws1gt_4,31,113,194).  visible(mws1gt_4,194). happensAt(walking(mws1gt_4),194).
orientation(mws1gt_1,155,195). coords(mws1gt_1,143,135,195).  visible(mws1gt_1,195). happensAt(walking(mws1gt_1),195). orientation(mws1gt_4,0,195). coords(mws1gt_4,31,113,195).  visible(mws1gt_4,195). happensAt(walking(mws1gt_4),195).
orientation(mws1gt_1,155,196). coords(mws1gt_1,143,135,196).  visible(mws1gt_1,196). happensAt(walking(mws1gt_1),196). orientation(mws1gt_4,0,196). coords(mws1gt_4,32,113,196).  visible(mws1gt_4,196). happensAt(walking(mws1gt_4),196).
orientation(mws1gt_1,155,197). coords(mws1gt_1,142,135,197).  visible(mws1gt_1,197). happensAt(walking(mws1gt_1),197). orientation(mws1gt_4,0,197). coords(mws1gt_4,32,112,197).  visible(mws1gt_4,197). happensAt(walking(mws1gt_4),197).
orientation(mws1gt_1,155,198). coords(mws1gt_1,142,134,198).  visible(mws1gt_1,198). happensAt(walking(mws1gt_1),198). orientation(mws1gt_4,0,198). coords(mws1gt_4,32,112,198).  visible(mws1gt_4,198). happensAt(walking(mws1gt_4),198).
#show holdsAt/2.