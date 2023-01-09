
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

	orientation(ms3ggt_0,149,182). coords(ms3ggt_0,66,135,182).  visible(ms3ggt_0,182). happensAt(walking(ms3ggt_0),182). orientation(ms3ggt_1,162,182). coords(ms3ggt_1,35,162,182).  visible(ms3ggt_1,182). happensAt(walking(ms3ggt_1),182).
orientation(ms3ggt_0,149,183). coords(ms3ggt_0,66,135,183).  visible(ms3ggt_0,183). happensAt(walking(ms3ggt_0),183). orientation(ms3ggt_1,162,183). coords(ms3ggt_1,35,162,183).  visible(ms3ggt_1,183). happensAt(walking(ms3ggt_1),183).
orientation(ms3ggt_0,149,184). coords(ms3ggt_0,65,136,184).  visible(ms3ggt_0,184). happensAt(walking(ms3ggt_0),184). orientation(ms3ggt_1,162,184). coords(ms3ggt_1,35,162,184).  visible(ms3ggt_1,184). happensAt(walking(ms3ggt_1),184).
orientation(ms3ggt_0,149,185). coords(ms3ggt_0,65,138,185).  visible(ms3ggt_0,185). happensAt(walking(ms3ggt_0),185). orientation(ms3ggt_1,162,185). coords(ms3ggt_1,35,162,185).  visible(ms3ggt_1,185). happensAt(walking(ms3ggt_1),185).
orientation(ms3ggt_0,149,186). coords(ms3ggt_0,65,138,186).  visible(ms3ggt_0,186). happensAt(walking(ms3ggt_0),186). orientation(ms3ggt_1,162,186). coords(ms3ggt_1,35,162,186).  visible(ms3ggt_1,186). happensAt(walking(ms3ggt_1),186).
orientation(ms3ggt_0,149,187). coords(ms3ggt_0,65,140,187).  visible(ms3ggt_0,187). happensAt(walking(ms3ggt_0),187). orientation(ms3ggt_1,162,187). coords(ms3ggt_1,35,162,187).  visible(ms3ggt_1,187). happensAt(walking(ms3ggt_1),187).
orientation(ms3ggt_0,149,188). coords(ms3ggt_0,65,140,188).  visible(ms3ggt_0,188). happensAt(walking(ms3ggt_0),188). orientation(ms3ggt_1,164,188). coords(ms3ggt_1,36,162,188).  visible(ms3ggt_1,188). happensAt(walking(ms3ggt_1),188).
orientation(ms3ggt_0,149,189). coords(ms3ggt_0,65,141,189).  visible(ms3ggt_0,189). happensAt(walking(ms3ggt_0),189). orientation(ms3ggt_1,164,189). coords(ms3ggt_1,36,162,189).  visible(ms3ggt_1,189). happensAt(walking(ms3ggt_1),189).
orientation(ms3ggt_0,149,190). coords(ms3ggt_0,64,141,190).  visible(ms3ggt_0,190). happensAt(walking(ms3ggt_0),190). orientation(ms3ggt_1,164,190). coords(ms3ggt_1,36,162,190).  visible(ms3ggt_1,190). happensAt(walking(ms3ggt_1),190).
orientation(ms3ggt_0,149,191). coords(ms3ggt_0,64,142,191).  visible(ms3ggt_0,191). happensAt(walking(ms3ggt_0),191). orientation(ms3ggt_1,164,191). coords(ms3ggt_1,36,162,191).  visible(ms3ggt_1,191). happensAt(walking(ms3ggt_1),191).
orientation(ms3ggt_0,149,192). coords(ms3ggt_0,64,142,192).  visible(ms3ggt_0,192). happensAt(walking(ms3ggt_0),192). orientation(ms3ggt_1,154,192). coords(ms3ggt_1,37,164,192).  visible(ms3ggt_1,192). happensAt(walking(ms3ggt_1),192).
orientation(ms3ggt_0,149,193). coords(ms3ggt_0,64,142,193).  visible(ms3ggt_0,193). happensAt(walking(ms3ggt_0),193). orientation(ms3ggt_1,154,193). coords(ms3ggt_1,37,164,193).  visible(ms3ggt_1,193). happensAt(walking(ms3ggt_1),193).
orientation(ms3ggt_0,157,194). coords(ms3ggt_0,63,142,194).  visible(ms3ggt_0,194). happensAt(walking(ms3ggt_0),194). orientation(ms3ggt_1,154,194). coords(ms3ggt_1,37,164,194).  visible(ms3ggt_1,194). happensAt(walking(ms3ggt_1),194).
orientation(ms3ggt_0,157,195). coords(ms3ggt_0,62,143,195).  visible(ms3ggt_0,195). happensAt(walking(ms3ggt_0),195). orientation(ms3ggt_1,154,195). coords(ms3ggt_1,37,164,195).  visible(ms3ggt_1,195). happensAt(walking(ms3ggt_1),195).
orientation(ms3ggt_0,157,196). coords(ms3ggt_0,62,143,196).  visible(ms3ggt_0,196). happensAt(walking(ms3ggt_0),196). orientation(ms3ggt_1,154,196). coords(ms3ggt_1,37,164,196).  visible(ms3ggt_1,196). happensAt(walking(ms3ggt_1),196).
orientation(ms3ggt_0,157,197). coords(ms3ggt_0,62,144,197).  visible(ms3ggt_0,197). happensAt(walking(ms3ggt_0),197). orientation(ms3ggt_1,154,197). coords(ms3ggt_1,37,164,197).  visible(ms3ggt_1,197). happensAt(walking(ms3ggt_1),197).
orientation(ms3ggt_0,145,198). coords(ms3ggt_0,61,148,198).  visible(ms3ggt_0,198). happensAt(walking(ms3ggt_0),198). orientation(ms3ggt_1,154,198). coords(ms3ggt_1,37,164,198).  visible(ms3ggt_1,198). happensAt(walking(ms3ggt_1),198).
orientation(ms3ggt_0,145,199). coords(ms3ggt_0,61,148,199).  visible(ms3ggt_0,199). happensAt(walking(ms3ggt_0),199). orientation(ms3ggt_1,154,199). coords(ms3ggt_1,37,164,199).  visible(ms3ggt_1,199). happensAt(walking(ms3ggt_1),199).
orientation(ms3ggt_0,145,200). coords(ms3ggt_0,61,148,200).  visible(ms3ggt_0,200). happensAt(walking(ms3ggt_0),200). orientation(ms3ggt_1,154,200). coords(ms3ggt_1,37,164,200).  visible(ms3ggt_1,200). happensAt(walking(ms3ggt_1),200).
orientation(ms3ggt_0,145,201). coords(ms3ggt_0,61,148,201).  visible(ms3ggt_0,201). happensAt(walking(ms3ggt_0),201). orientation(ms3ggt_1,154,201). coords(ms3ggt_1,37,164,201).  visible(ms3ggt_1,201). happensAt(walking(ms3ggt_1),201).
orientation(ms3ggt_0,145,202). coords(ms3ggt_0,61,148,202).  visible(ms3ggt_0,202). happensAt(walking(ms3ggt_0),202). orientation(ms3ggt_1,154,202). coords(ms3ggt_1,37,164,202).  visible(ms3ggt_1,202). happensAt(walking(ms3ggt_1),202).
orientation(ms3ggt_0,145,203). coords(ms3ggt_0,61,150,203).  visible(ms3ggt_0,203). happensAt(walking(ms3ggt_0),203). orientation(ms3ggt_1,154,203). coords(ms3ggt_1,37,164,203).  visible(ms3ggt_1,203). happensAt(walking(ms3ggt_1),203).
orientation(ms3ggt_0,145,204). coords(ms3ggt_0,61,151,204).  visible(ms3ggt_0,204). happensAt(walking(ms3ggt_0),204). orientation(ms3ggt_1,154,204). coords(ms3ggt_1,38,165,204).  visible(ms3ggt_1,204). happensAt(walking(ms3ggt_1),204).
orientation(ms3ggt_0,145,205). coords(ms3ggt_0,61,151,205).  visible(ms3ggt_0,205). happensAt(walking(ms3ggt_0),205). orientation(ms3ggt_1,154,205). coords(ms3ggt_1,39,165,205).  visible(ms3ggt_1,205). happensAt(walking(ms3ggt_1),205).
orientation(ms3ggt_0,145,206). coords(ms3ggt_0,60,152,206).  visible(ms3ggt_0,206). happensAt(walking(ms3ggt_0),206). orientation(ms3ggt_1,154,206). coords(ms3ggt_1,41,165,206).  visible(ms3ggt_1,206). happensAt(walking(ms3ggt_1),206).
#show holdsAt/2.