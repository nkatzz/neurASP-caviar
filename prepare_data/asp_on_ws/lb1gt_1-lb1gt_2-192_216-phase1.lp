
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

	orientation(lb1gt_1,151,192). coords(lb1gt_1,112,151,192).  visible(lb1gt_1,192). happensAt(walking(lb1gt_1),192). orientation(lb1gt_2,144,192). coords(lb1gt_2,120,115,192).  visible(lb1gt_2,192). happensAt(walking(lb1gt_2),192).
orientation(lb1gt_1,151,193). coords(lb1gt_1,112,152,193).  visible(lb1gt_1,193). happensAt(walking(lb1gt_1),193). orientation(lb1gt_2,144,193). coords(lb1gt_2,121,115,193).  visible(lb1gt_2,193). happensAt(walking(lb1gt_2),193).
orientation(lb1gt_1,151,194). coords(lb1gt_1,113,153,194).  visible(lb1gt_1,194). happensAt(walking(lb1gt_1),194). orientation(lb1gt_2,144,194). coords(lb1gt_2,122,115,194).  visible(lb1gt_2,194). happensAt(walking(lb1gt_2),194).
orientation(lb1gt_1,151,195). coords(lb1gt_1,113,154,195).  visible(lb1gt_1,195). happensAt(walking(lb1gt_1),195). orientation(lb1gt_2,144,195). coords(lb1gt_2,124,115,195).  visible(lb1gt_2,195). happensAt(walking(lb1gt_2),195).
orientation(lb1gt_1,151,196). coords(lb1gt_1,115,155,196).  visible(lb1gt_1,196). happensAt(walking(lb1gt_1),196). orientation(lb1gt_2,144,196). coords(lb1gt_2,125,115,196).  visible(lb1gt_2,196). happensAt(walking(lb1gt_2),196).
orientation(lb1gt_1,151,197). coords(lb1gt_1,116,156,197).  visible(lb1gt_1,197). happensAt(walking(lb1gt_1),197). orientation(lb1gt_2,144,197). coords(lb1gt_2,126,115,197).  visible(lb1gt_2,197). happensAt(walking(lb1gt_2),197).
orientation(lb1gt_1,151,198). coords(lb1gt_1,118,156,198).  visible(lb1gt_1,198). happensAt(walking(lb1gt_1),198). orientation(lb1gt_2,144,198). coords(lb1gt_2,127,115,198).  visible(lb1gt_2,198). happensAt(walking(lb1gt_2),198).
orientation(lb1gt_1,151,199). coords(lb1gt_1,118,156,199).  visible(lb1gt_1,199). happensAt(walking(lb1gt_1),199). orientation(lb1gt_2,144,199). coords(lb1gt_2,128,115,199).  visible(lb1gt_2,199). happensAt(walking(lb1gt_2),199).
orientation(lb1gt_1,151,200). coords(lb1gt_1,119,157,200).  visible(lb1gt_1,200). happensAt(walking(lb1gt_1),200). orientation(lb1gt_2,144,200). coords(lb1gt_2,129,115,200).  visible(lb1gt_2,200). happensAt(walking(lb1gt_2),200).
orientation(lb1gt_1,151,201). coords(lb1gt_1,120,157,201).  visible(lb1gt_1,201). happensAt(walking(lb1gt_1),201). orientation(lb1gt_2,144,201). coords(lb1gt_2,129,115,201).  visible(lb1gt_2,201). happensAt(walking(lb1gt_2),201).
orientation(lb1gt_1,151,202). coords(lb1gt_1,120,157,202).  visible(lb1gt_1,202). happensAt(walking(lb1gt_1),202). orientation(lb1gt_2,144,202). coords(lb1gt_2,130,115,202).  visible(lb1gt_2,202). happensAt(walking(lb1gt_2),202).
orientation(lb1gt_1,151,203). coords(lb1gt_1,121,158,203).  visible(lb1gt_1,203). happensAt(walking(lb1gt_1),203). orientation(lb1gt_2,144,203). coords(lb1gt_2,130,115,203).  visible(lb1gt_2,203). happensAt(walking(lb1gt_2),203).
orientation(lb1gt_1,151,204). coords(lb1gt_1,121,158,204).  visible(lb1gt_1,204). happensAt(walking(lb1gt_1),204). orientation(lb1gt_2,147,204). coords(lb1gt_2,130,116,204).  visible(lb1gt_2,204). happensAt(walking(lb1gt_2),204).
orientation(lb1gt_1,151,205). coords(lb1gt_1,121,158,205).  visible(lb1gt_1,205). happensAt(walking(lb1gt_1),205). orientation(lb1gt_2,147,205). coords(lb1gt_2,131,116,205).  visible(lb1gt_2,205). happensAt(walking(lb1gt_2),205).
orientation(lb1gt_1,151,206). coords(lb1gt_1,121,158,206).  visible(lb1gt_1,206). happensAt(walking(lb1gt_1),206). orientation(lb1gt_2,147,206). coords(lb1gt_2,131,116,206).  visible(lb1gt_2,206). happensAt(walking(lb1gt_2),206).
orientation(lb1gt_1,151,207). coords(lb1gt_1,121,158,207).  visible(lb1gt_1,207). happensAt(walking(lb1gt_1),207). orientation(lb1gt_2,147,207). coords(lb1gt_2,132,116,207).  visible(lb1gt_2,207). happensAt(walking(lb1gt_2),207).
orientation(lb1gt_1,151,208). coords(lb1gt_1,121,158,208).  visible(lb1gt_1,208). happensAt(walking(lb1gt_1),208). orientation(lb1gt_2,147,208). coords(lb1gt_2,132,116,208).  visible(lb1gt_2,208). happensAt(walking(lb1gt_2),208).
orientation(lb1gt_1,151,209). coords(lb1gt_1,121,159,209).  visible(lb1gt_1,209). happensAt(walking(lb1gt_1),209). orientation(lb1gt_2,147,209). coords(lb1gt_2,132,116,209).  visible(lb1gt_2,209). happensAt(walking(lb1gt_2),209).
orientation(lb1gt_1,151,210). coords(lb1gt_1,122,159,210).  visible(lb1gt_1,210). happensAt(walking(lb1gt_1),210). orientation(lb1gt_2,147,210). coords(lb1gt_2,133,117,210).  visible(lb1gt_2,210). happensAt(walking(lb1gt_2),210).
orientation(lb1gt_1,151,211). coords(lb1gt_1,122,159,211).  visible(lb1gt_1,211). happensAt(walking(lb1gt_1),211). orientation(lb1gt_2,147,211). coords(lb1gt_2,133,117,211).  visible(lb1gt_2,211). happensAt(walking(lb1gt_2),211).
orientation(lb1gt_1,151,212). coords(lb1gt_1,123,159,212).  visible(lb1gt_1,212). happensAt(walking(lb1gt_1),212). orientation(lb1gt_2,147,212). coords(lb1gt_2,133,118,212).  visible(lb1gt_2,212). happensAt(walking(lb1gt_2),212).
orientation(lb1gt_1,151,213). coords(lb1gt_1,124,159,213).  visible(lb1gt_1,213). happensAt(walking(lb1gt_1),213). orientation(lb1gt_2,147,213). coords(lb1gt_2,134,119,213).  visible(lb1gt_2,213). happensAt(walking(lb1gt_2),213).
orientation(lb1gt_1,151,214). coords(lb1gt_1,126,159,214).  visible(lb1gt_1,214). happensAt(walking(lb1gt_1),214). orientation(lb1gt_2,147,214). coords(lb1gt_2,135,120,214).  visible(lb1gt_2,214). happensAt(walking(lb1gt_2),214).
orientation(lb1gt_1,151,215). coords(lb1gt_1,127,159,215).  visible(lb1gt_1,215). happensAt(walking(lb1gt_1),215). orientation(lb1gt_2,147,215). coords(lb1gt_2,135,120,215).  visible(lb1gt_2,215). happensAt(walking(lb1gt_2),215).
orientation(lb1gt_1,151,216). coords(lb1gt_1,129,159,216).  visible(lb1gt_1,216). happensAt(walking(lb1gt_1),216). orientation(lb1gt_2,147,216). coords(lb1gt_2,136,120,216).  visible(lb1gt_2,216). happensAt(walking(lb1gt_2),216).
#show holdsAt/2.