
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

	orientation(mc1gt_0,147,184). coords(mc1gt_0,123,112,184).  visible(mc1gt_0,184). happensAt(walking(mc1gt_0),184). orientation(mc1gt_1,145,184). coords(mc1gt_1,146,92,184).  visible(mc1gt_1,184). happensAt(walking(mc1gt_1),184).
orientation(mc1gt_0,147,185). coords(mc1gt_0,123,113,185).  visible(mc1gt_0,185). happensAt(walking(mc1gt_0),185). orientation(mc1gt_1,145,185). coords(mc1gt_1,147,94,185).  visible(mc1gt_1,185). happensAt(walking(mc1gt_1),185).
orientation(mc1gt_0,147,186). coords(mc1gt_0,123,114,186).  visible(mc1gt_0,186). happensAt(walking(mc1gt_0),186). orientation(mc1gt_1,145,186). coords(mc1gt_1,148,94,186).  visible(mc1gt_1,186). happensAt(walking(mc1gt_1),186).
orientation(mc1gt_0,147,187). coords(mc1gt_0,124,114,187).  visible(mc1gt_0,187). happensAt(walking(mc1gt_0),187). orientation(mc1gt_1,145,187). coords(mc1gt_1,149,95,187).  visible(mc1gt_1,187). happensAt(walking(mc1gt_1),187).
orientation(mc1gt_0,147,188). coords(mc1gt_0,126,115,188).  visible(mc1gt_0,188). happensAt(walking(mc1gt_0),188). orientation(mc1gt_1,143,188). coords(mc1gt_1,150,96,188).  visible(mc1gt_1,188). happensAt(walking(mc1gt_1),188).
orientation(mc1gt_0,147,189). coords(mc1gt_0,127,115,189).  visible(mc1gt_0,189). happensAt(walking(mc1gt_0),189). orientation(mc1gt_1,141,189). coords(mc1gt_1,151,96,189).  visible(mc1gt_1,189). happensAt(walking(mc1gt_1),189).
orientation(mc1gt_0,147,190). coords(mc1gt_0,128,116,190).  visible(mc1gt_0,190). happensAt(walking(mc1gt_0),190). orientation(mc1gt_1,139,190). coords(mc1gt_1,151,97,190).  visible(mc1gt_1,190). happensAt(walking(mc1gt_1),190).
orientation(mc1gt_0,147,191). coords(mc1gt_0,129,116,191).  visible(mc1gt_0,191). happensAt(walking(mc1gt_0),191). orientation(mc1gt_1,139,191). coords(mc1gt_1,152,97,191).  visible(mc1gt_1,191). happensAt(walking(mc1gt_1),191).
orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). happensAt(walking(mc1gt_0),192). orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192). happensAt(walking(mc1gt_1),192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). happensAt(walking(mc1gt_0),193). orientation(mc1gt_1,140,193). coords(mc1gt_1,152,97,193).  visible(mc1gt_1,193). happensAt(walking(mc1gt_1),193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). happensAt(walking(mc1gt_0),194). orientation(mc1gt_1,140,194). coords(mc1gt_1,152,97,194).  visible(mc1gt_1,194). happensAt(walking(mc1gt_1),194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). happensAt(walking(mc1gt_0),195). orientation(mc1gt_1,140,195). coords(mc1gt_1,152,97,195).  visible(mc1gt_1,195). happensAt(walking(mc1gt_1),195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). happensAt(walking(mc1gt_0),196). orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196). happensAt(walking(mc1gt_1),196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). happensAt(walking(mc1gt_0),197). orientation(mc1gt_1,140,197). coords(mc1gt_1,153,100,197).  visible(mc1gt_1,197). happensAt(walking(mc1gt_1),197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). happensAt(walking(mc1gt_0),198). orientation(mc1gt_1,139,198). coords(mc1gt_1,154,101,198).  visible(mc1gt_1,198). happensAt(walking(mc1gt_1),198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). happensAt(walking(mc1gt_0),199). orientation(mc1gt_1,139,199). coords(mc1gt_1,155,103,199).  visible(mc1gt_1,199). happensAt(walking(mc1gt_1),199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). happensAt(walking(mc1gt_0),200). orientation(mc1gt_1,139,200). coords(mc1gt_1,157,105,200).  visible(mc1gt_1,200). happensAt(walking(mc1gt_1),200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). happensAt(walking(mc1gt_0),201). orientation(mc1gt_1,137,201). coords(mc1gt_1,159,106,201).  visible(mc1gt_1,201). happensAt(walking(mc1gt_1),201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,138,118,202).  visible(mc1gt_0,202). happensAt(walking(mc1gt_0),202). orientation(mc1gt_1,137,202). coords(mc1gt_1,160,107,202).  visible(mc1gt_1,202). happensAt(walking(mc1gt_1),202).
orientation(mc1gt_0,147,203). coords(mc1gt_0,139,118,203).  visible(mc1gt_0,203). happensAt(walking(mc1gt_0),203). orientation(mc1gt_1,137,203). coords(mc1gt_1,161,108,203).  visible(mc1gt_1,203). happensAt(walking(mc1gt_1),203).
orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). happensAt(walking(mc1gt_0),204). orientation(mc1gt_1,137,204). coords(mc1gt_1,161,109,204).  visible(mc1gt_1,204). happensAt(walking(mc1gt_1),204).
orientation(mc1gt_0,147,205). coords(mc1gt_0,141,119,205).  visible(mc1gt_0,205). happensAt(walking(mc1gt_0),205). orientation(mc1gt_1,137,205). coords(mc1gt_1,162,109,205).  visible(mc1gt_1,205). happensAt(walking(mc1gt_1),205).
orientation(mc1gt_0,147,206). coords(mc1gt_0,141,119,206).  visible(mc1gt_0,206). happensAt(walking(mc1gt_0),206). orientation(mc1gt_1,140,206). coords(mc1gt_1,162,110,206).  visible(mc1gt_1,206). happensAt(walking(mc1gt_1),206).
orientation(mc1gt_0,147,207). coords(mc1gt_0,142,119,207).  visible(mc1gt_0,207). happensAt(walking(mc1gt_0),207). orientation(mc1gt_1,140,207). coords(mc1gt_1,162,110,207).  visible(mc1gt_1,207). happensAt(walking(mc1gt_1),207).
orientation(mc1gt_0,147,208). coords(mc1gt_0,142,119,208).  visible(mc1gt_0,208). happensAt(walking(mc1gt_0),208). orientation(mc1gt_1,140,208). coords(mc1gt_1,162,110,208).  visible(mc1gt_1,208). happensAt(walking(mc1gt_1),208).
#show holdsAt/2.