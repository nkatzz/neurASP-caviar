
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

	orientation(lbbcgt_1,137,174). coords(lbbcgt_1,5,84,174).  visible(lbbcgt_1,174). happensAt(walking(lbbcgt_1),174). orientation(lbbcgt_2,101,174). coords(lbbcgt_2,268,6,174).  visible(lbbcgt_2,174). happensAt(walking(lbbcgt_2),174).
orientation(lbbcgt_1,137,175). coords(lbbcgt_1,4,85,175).  visible(lbbcgt_1,175). happensAt(walking(lbbcgt_1),175). orientation(lbbcgt_2,98,175). coords(lbbcgt_2,268,7,175).  visible(lbbcgt_2,175). happensAt(walking(lbbcgt_2),175).
orientation(lbbcgt_1,137,176). coords(lbbcgt_1,4,87,176).  visible(lbbcgt_1,176). happensAt(walking(lbbcgt_1),176). orientation(lbbcgt_2,96,176). coords(lbbcgt_2,268,7,176).  visible(lbbcgt_2,176). happensAt(walking(lbbcgt_2),176).
orientation(lbbcgt_1,137,177). coords(lbbcgt_1,4,87,177).  visible(lbbcgt_1,177). happensAt(walking(lbbcgt_1),177). orientation(lbbcgt_2,96,177). coords(lbbcgt_2,268,7,177).  visible(lbbcgt_2,177). happensAt(walking(lbbcgt_2),177).
orientation(lbbcgt_1,137,178). coords(lbbcgt_1,4,88,178).  visible(lbbcgt_1,178). happensAt(walking(lbbcgt_1),178). orientation(lbbcgt_2,105,178). coords(lbbcgt_2,263,7,178).  visible(lbbcgt_2,178). happensAt(walking(lbbcgt_2),178).
orientation(lbbcgt_1,137,179). coords(lbbcgt_1,4,88,179).  visible(lbbcgt_1,179). happensAt(walking(lbbcgt_1),179). orientation(lbbcgt_2,111,179). coords(lbbcgt_2,263,7,179).  visible(lbbcgt_2,179). happensAt(walking(lbbcgt_2),179).
orientation(lbbcgt_1,148,180). coords(lbbcgt_1,4,88,180).  visible(lbbcgt_1,180). happensAt(walking(lbbcgt_1),180). orientation(lbbcgt_2,109,180). coords(lbbcgt_2,263,7,180).  visible(lbbcgt_2,180). happensAt(walking(lbbcgt_2),180).
orientation(lbbcgt_1,148,181). coords(lbbcgt_1,4,88,181).  visible(lbbcgt_1,181). happensAt(walking(lbbcgt_1),181). orientation(lbbcgt_2,104,181). coords(lbbcgt_2,263,7,181).  visible(lbbcgt_2,181). happensAt(walking(lbbcgt_2),181).
orientation(lbbcgt_1,148,182). coords(lbbcgt_1,4,90,182).  visible(lbbcgt_1,182). happensAt(walking(lbbcgt_1),182). orientation(lbbcgt_2,104,182). coords(lbbcgt_2,263,9,182).  visible(lbbcgt_2,182). happensAt(walking(lbbcgt_2),182).
orientation(lbbcgt_1,148,183). coords(lbbcgt_1,4,90,183).  visible(lbbcgt_1,183). happensAt(walking(lbbcgt_1),183). orientation(lbbcgt_2,104,183). coords(lbbcgt_2,263,10,183).  visible(lbbcgt_2,183). happensAt(walking(lbbcgt_2),183).
orientation(lbbcgt_1,148,184). coords(lbbcgt_1,4,92,184).  visible(lbbcgt_1,184). happensAt(walking(lbbcgt_1),184). orientation(lbbcgt_2,104,184). coords(lbbcgt_2,261,11,184).  visible(lbbcgt_2,184). happensAt(walking(lbbcgt_2),184).
orientation(lbbcgt_1,148,185). coords(lbbcgt_1,4,93,185).  visible(lbbcgt_1,185). happensAt(walking(lbbcgt_1),185). orientation(lbbcgt_2,104,185). coords(lbbcgt_2,261,12,185).  visible(lbbcgt_2,185). happensAt(walking(lbbcgt_2),185).
orientation(lbbcgt_1,148,186). coords(lbbcgt_1,4,94,186).  visible(lbbcgt_1,186). happensAt(walking(lbbcgt_1),186). orientation(lbbcgt_2,115,186). coords(lbbcgt_2,261,13,186).  visible(lbbcgt_2,186). happensAt(walking(lbbcgt_2),186).
orientation(lbbcgt_1,148,187). coords(lbbcgt_1,3,94,187).  visible(lbbcgt_1,187). happensAt(walking(lbbcgt_1),187). orientation(lbbcgt_2,115,187). coords(lbbcgt_2,259,13,187).  visible(lbbcgt_2,187). happensAt(walking(lbbcgt_2),187).
orientation(lbbcgt_1,148,188). coords(lbbcgt_1,3,94,188).  visible(lbbcgt_1,188). happensAt(walking(lbbcgt_1),188). orientation(lbbcgt_2,115,188). coords(lbbcgt_2,258,13,188).  visible(lbbcgt_2,188). happensAt(walking(lbbcgt_2),188).
orientation(lbbcgt_1,148,189). coords(lbbcgt_1,3,94,189).  visible(lbbcgt_1,189). happensAt(walking(lbbcgt_1),189). orientation(lbbcgt_2,115,189). coords(lbbcgt_2,256,13,189).  visible(lbbcgt_2,189). happensAt(walking(lbbcgt_2),189).
orientation(lbbcgt_1,148,190). coords(lbbcgt_1,3,95,190).  visible(lbbcgt_1,190). happensAt(walking(lbbcgt_1),190). orientation(lbbcgt_2,115,190). coords(lbbcgt_2,255,13,190).  visible(lbbcgt_2,190). happensAt(walking(lbbcgt_2),190).
orientation(lbbcgt_1,148,191). coords(lbbcgt_1,3,95,191).  visible(lbbcgt_1,191). happensAt(walking(lbbcgt_1),191). orientation(lbbcgt_2,118,191). coords(lbbcgt_2,254,13,191).  visible(lbbcgt_2,191). happensAt(walking(lbbcgt_2),191).
orientation(lbbcgt_1,148,192). coords(lbbcgt_1,3,96,192).  visible(lbbcgt_1,192). happensAt(walking(lbbcgt_1),192). orientation(lbbcgt_2,118,192). coords(lbbcgt_2,252,13,192).  visible(lbbcgt_2,192). happensAt(walking(lbbcgt_2),192).
orientation(lbbcgt_1,148,193). coords(lbbcgt_1,3,96,193).  visible(lbbcgt_1,193). happensAt(walking(lbbcgt_1),193). orientation(lbbcgt_2,118,193). coords(lbbcgt_2,251,13,193).  visible(lbbcgt_2,193). happensAt(walking(lbbcgt_2),193).
orientation(lbbcgt_1,148,194). coords(lbbcgt_1,3,96,194).  visible(lbbcgt_1,194). happensAt(walking(lbbcgt_1),194). orientation(lbbcgt_2,118,194). coords(lbbcgt_2,250,14,194).  visible(lbbcgt_2,194). happensAt(walking(lbbcgt_2),194).
orientation(lbbcgt_1,148,195). coords(lbbcgt_1,3,96,195).  visible(lbbcgt_1,195). happensAt(walking(lbbcgt_1),195). orientation(lbbcgt_2,118,195). coords(lbbcgt_2,248,15,195).  visible(lbbcgt_2,195). happensAt(walking(lbbcgt_2),195).
orientation(lbbcgt_1,148,196). coords(lbbcgt_1,3,97,196).  visible(lbbcgt_1,196). happensAt(walking(lbbcgt_1),196). orientation(lbbcgt_2,118,196). coords(lbbcgt_2,246,17,196).  visible(lbbcgt_2,196). happensAt(walking(lbbcgt_2),196).
orientation(lbbcgt_1,148,197). coords(lbbcgt_1,3,97,197).  visible(lbbcgt_1,197). happensAt(walking(lbbcgt_1),197). orientation(lbbcgt_2,118,197). coords(lbbcgt_2,246,18,197).  visible(lbbcgt_2,197). happensAt(walking(lbbcgt_2),197).
orientation(lbbcgt_1,148,198). coords(lbbcgt_1,3,98,198).  happensAt(disappear(lbbcgt_1),198). happensAt(walking(lbbcgt_1),198). orientation(lbbcgt_2,118,198). coords(lbbcgt_2,245,19,198).  visible(lbbcgt_2,198). happensAt(walking(lbbcgt_2),198).
#show holdsAt/2.