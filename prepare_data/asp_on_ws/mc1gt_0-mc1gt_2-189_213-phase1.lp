
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

	orientation(mc1gt_0,147,189). coords(mc1gt_0,127,115,189).  visible(mc1gt_0,189). happensAt(walking(mc1gt_0),189). orientation(mc1gt_2,147,189). coords(mc1gt_2,95,115,189).  visible(mc1gt_2,189). happensAt(walking(mc1gt_2),189).
orientation(mc1gt_0,147,190). coords(mc1gt_0,128,116,190).  visible(mc1gt_0,190). happensAt(walking(mc1gt_0),190). orientation(mc1gt_2,148,190). coords(mc1gt_2,97,115,190).  visible(mc1gt_2,190). happensAt(walking(mc1gt_2),190).
orientation(mc1gt_0,147,191). coords(mc1gt_0,129,116,191).  visible(mc1gt_0,191). happensAt(walking(mc1gt_0),191). orientation(mc1gt_2,149,191). coords(mc1gt_2,98,116,191).  visible(mc1gt_2,191). happensAt(walking(mc1gt_2),191).
orientation(mc1gt_0,147,192). coords(mc1gt_0,130,116,192).  visible(mc1gt_0,192). happensAt(walking(mc1gt_0),192). orientation(mc1gt_2,149,192). coords(mc1gt_2,99,116,192).  visible(mc1gt_2,192). happensAt(walking(mc1gt_2),192).
orientation(mc1gt_0,147,193). coords(mc1gt_0,130,117,193).  visible(mc1gt_0,193). happensAt(walking(mc1gt_0),193). orientation(mc1gt_2,149,193). coords(mc1gt_2,100,116,193).  visible(mc1gt_2,193). happensAt(walking(mc1gt_2),193).
orientation(mc1gt_0,147,194). coords(mc1gt_0,130,117,194).  visible(mc1gt_0,194). happensAt(walking(mc1gt_0),194). orientation(mc1gt_2,149,194). coords(mc1gt_2,101,117,194).  visible(mc1gt_2,194). happensAt(walking(mc1gt_2),194).
orientation(mc1gt_0,147,195). coords(mc1gt_0,131,117,195).  visible(mc1gt_0,195). happensAt(walking(mc1gt_0),195). orientation(mc1gt_2,148,195). coords(mc1gt_2,101,117,195).  visible(mc1gt_2,195). happensAt(walking(mc1gt_2),195).
orientation(mc1gt_0,147,196). coords(mc1gt_0,131,117,196).  visible(mc1gt_0,196). happensAt(walking(mc1gt_0),196). orientation(mc1gt_2,148,196). coords(mc1gt_2,102,117,196).  visible(mc1gt_2,196). happensAt(walking(mc1gt_2),196).
orientation(mc1gt_0,147,197). coords(mc1gt_0,131,117,197).  visible(mc1gt_0,197). happensAt(walking(mc1gt_0),197). orientation(mc1gt_2,148,197). coords(mc1gt_2,102,117,197).  visible(mc1gt_2,197). happensAt(walking(mc1gt_2),197).
orientation(mc1gt_0,147,198). coords(mc1gt_0,131,117,198).  visible(mc1gt_0,198). happensAt(walking(mc1gt_0),198). orientation(mc1gt_2,147,198). coords(mc1gt_2,102,117,198).  visible(mc1gt_2,198). happensAt(walking(mc1gt_2),198).
orientation(mc1gt_0,147,199). coords(mc1gt_0,132,117,199).  visible(mc1gt_0,199). happensAt(walking(mc1gt_0),199). orientation(mc1gt_2,147,199). coords(mc1gt_2,102,117,199).  visible(mc1gt_2,199). happensAt(walking(mc1gt_2),199).
orientation(mc1gt_0,147,200). coords(mc1gt_0,134,117,200).  visible(mc1gt_0,200). happensAt(walking(mc1gt_0),200). orientation(mc1gt_2,147,200). coords(mc1gt_2,103,118,200).  visible(mc1gt_2,200). happensAt(walking(mc1gt_2),200).
orientation(mc1gt_0,147,201). coords(mc1gt_0,136,118,201).  visible(mc1gt_0,201). happensAt(walking(mc1gt_0),201). orientation(mc1gt_2,147,201). coords(mc1gt_2,103,118,201).  visible(mc1gt_2,201). happensAt(walking(mc1gt_2),201).
orientation(mc1gt_0,147,202). coords(mc1gt_0,138,118,202).  visible(mc1gt_0,202). happensAt(walking(mc1gt_0),202). orientation(mc1gt_2,148,202). coords(mc1gt_2,103,119,202).  visible(mc1gt_2,202). happensAt(walking(mc1gt_2),202).
orientation(mc1gt_0,147,203). coords(mc1gt_0,139,118,203).  visible(mc1gt_0,203). happensAt(walking(mc1gt_0),203). orientation(mc1gt_2,148,203). coords(mc1gt_2,104,120,203).  visible(mc1gt_2,203). happensAt(walking(mc1gt_2),203).
orientation(mc1gt_0,147,204). coords(mc1gt_0,140,118,204).  visible(mc1gt_0,204). happensAt(walking(mc1gt_0),204). orientation(mc1gt_2,148,204). coords(mc1gt_2,106,121,204).  visible(mc1gt_2,204). happensAt(walking(mc1gt_2),204).
orientation(mc1gt_0,147,205). coords(mc1gt_0,141,119,205).  visible(mc1gt_0,205). happensAt(walking(mc1gt_0),205). orientation(mc1gt_2,148,205). coords(mc1gt_2,106,122,205).  visible(mc1gt_2,205). happensAt(walking(mc1gt_2),205).
orientation(mc1gt_0,147,206). coords(mc1gt_0,141,119,206).  visible(mc1gt_0,206). happensAt(walking(mc1gt_0),206). orientation(mc1gt_2,148,206). coords(mc1gt_2,108,122,206).  visible(mc1gt_2,206). happensAt(walking(mc1gt_2),206).
orientation(mc1gt_0,147,207). coords(mc1gt_0,142,119,207).  visible(mc1gt_0,207). happensAt(walking(mc1gt_0),207). orientation(mc1gt_2,148,207). coords(mc1gt_2,109,123,207).  visible(mc1gt_2,207). happensAt(walking(mc1gt_2),207).
orientation(mc1gt_0,147,208). coords(mc1gt_0,142,119,208).  visible(mc1gt_0,208). happensAt(walking(mc1gt_0),208). orientation(mc1gt_2,148,208). coords(mc1gt_2,109,123,208).  visible(mc1gt_2,208). happensAt(walking(mc1gt_2),208).
orientation(mc1gt_0,147,209). coords(mc1gt_0,142,119,209).  visible(mc1gt_0,209). happensAt(walking(mc1gt_0),209). orientation(mc1gt_2,148,209). coords(mc1gt_2,110,124,209).  visible(mc1gt_2,209). happensAt(walking(mc1gt_2),209).
orientation(mc1gt_0,147,210). coords(mc1gt_0,143,119,210).  visible(mc1gt_0,210). happensAt(walking(mc1gt_0),210). orientation(mc1gt_2,148,210). coords(mc1gt_2,110,124,210).  visible(mc1gt_2,210). happensAt(walking(mc1gt_2),210).
orientation(mc1gt_0,147,211). coords(mc1gt_0,143,120,211).  visible(mc1gt_0,211). happensAt(walking(mc1gt_0),211). orientation(mc1gt_2,148,211). coords(mc1gt_2,110,124,211).  visible(mc1gt_2,211). happensAt(walking(mc1gt_2),211).
orientation(mc1gt_0,147,212). coords(mc1gt_0,143,121,212).  visible(mc1gt_0,212). happensAt(walking(mc1gt_0),212). orientation(mc1gt_2,148,212). coords(mc1gt_2,111,124,212).  visible(mc1gt_2,212). happensAt(walking(mc1gt_2),212).
orientation(mc1gt_0,147,213). coords(mc1gt_0,144,122,213).  visible(mc1gt_0,213). happensAt(walking(mc1gt_0),213). orientation(mc1gt_2,148,213). coords(mc1gt_2,111,125,213).  visible(mc1gt_2,213). happensAt(walking(mc1gt_2),213).
#show holdsAt/2.