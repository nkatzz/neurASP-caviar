
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

	orientation(lb1gt_0,136,192). coords(lb1gt_0,166,33,192).  visible(lb1gt_0,192). happensAt(walking(lb1gt_0),192). orientation(lb1gt_1,151,192). coords(lb1gt_1,112,151,192).  visible(lb1gt_1,192). happensAt(walking(lb1gt_1),192).
orientation(lb1gt_0,136,193). coords(lb1gt_0,166,33,193).  visible(lb1gt_0,193). happensAt(walking(lb1gt_0),193). orientation(lb1gt_1,151,193). coords(lb1gt_1,112,152,193).  visible(lb1gt_1,193). happensAt(walking(lb1gt_1),193).
orientation(lb1gt_0,136,194). coords(lb1gt_0,166,33,194).  visible(lb1gt_0,194). happensAt(walking(lb1gt_0),194). orientation(lb1gt_1,151,194). coords(lb1gt_1,113,153,194).  visible(lb1gt_1,194). happensAt(walking(lb1gt_1),194).
orientation(lb1gt_0,136,195). coords(lb1gt_0,166,33,195).  visible(lb1gt_0,195). happensAt(walking(lb1gt_0),195). orientation(lb1gt_1,151,195). coords(lb1gt_1,113,154,195).  visible(lb1gt_1,195). happensAt(walking(lb1gt_1),195).
orientation(lb1gt_0,136,196). coords(lb1gt_0,166,32,196).  visible(lb1gt_0,196). happensAt(walking(lb1gt_0),196). orientation(lb1gt_1,151,196). coords(lb1gt_1,115,155,196).  visible(lb1gt_1,196). happensAt(walking(lb1gt_1),196).
orientation(lb1gt_0,136,197). coords(lb1gt_0,167,32,197).  visible(lb1gt_0,197). happensAt(walking(lb1gt_0),197). orientation(lb1gt_1,151,197). coords(lb1gt_1,116,156,197).  visible(lb1gt_1,197). happensAt(walking(lb1gt_1),197).
orientation(lb1gt_0,136,198). coords(lb1gt_0,167,32,198).  visible(lb1gt_0,198). happensAt(walking(lb1gt_0),198). orientation(lb1gt_1,151,198). coords(lb1gt_1,118,156,198).  visible(lb1gt_1,198). happensAt(walking(lb1gt_1),198).
orientation(lb1gt_0,136,199). coords(lb1gt_0,168,31,199).  visible(lb1gt_0,199). happensAt(walking(lb1gt_0),199). orientation(lb1gt_1,151,199). coords(lb1gt_1,118,156,199).  visible(lb1gt_1,199). happensAt(walking(lb1gt_1),199).
orientation(lb1gt_0,136,200). coords(lb1gt_0,168,31,200).  visible(lb1gt_0,200). happensAt(walking(lb1gt_0),200). orientation(lb1gt_1,151,200). coords(lb1gt_1,119,157,200).  visible(lb1gt_1,200). happensAt(walking(lb1gt_1),200).
orientation(lb1gt_0,136,201). coords(lb1gt_0,168,31,201).  visible(lb1gt_0,201). happensAt(walking(lb1gt_0),201). orientation(lb1gt_1,151,201). coords(lb1gt_1,120,157,201).  visible(lb1gt_1,201). happensAt(walking(lb1gt_1),201).
orientation(lb1gt_0,136,202). coords(lb1gt_0,168,31,202).  visible(lb1gt_0,202). happensAt(walking(lb1gt_0),202). orientation(lb1gt_1,151,202). coords(lb1gt_1,120,157,202).  visible(lb1gt_1,202). happensAt(walking(lb1gt_1),202).
orientation(lb1gt_0,136,203). coords(lb1gt_0,169,31,203).  visible(lb1gt_0,203). happensAt(walking(lb1gt_0),203). orientation(lb1gt_1,151,203). coords(lb1gt_1,121,158,203).  visible(lb1gt_1,203). happensAt(walking(lb1gt_1),203).
orientation(lb1gt_0,136,204). coords(lb1gt_0,169,30,204).  visible(lb1gt_0,204). happensAt(walking(lb1gt_0),204). orientation(lb1gt_1,151,204). coords(lb1gt_1,121,158,204).  visible(lb1gt_1,204). happensAt(walking(lb1gt_1),204).
orientation(lb1gt_0,136,205). coords(lb1gt_0,170,30,205).  visible(lb1gt_0,205). happensAt(walking(lb1gt_0),205). orientation(lb1gt_1,151,205). coords(lb1gt_1,121,158,205).  visible(lb1gt_1,205). happensAt(walking(lb1gt_1),205).
orientation(lb1gt_0,136,206). coords(lb1gt_0,170,29,206).  visible(lb1gt_0,206). happensAt(walking(lb1gt_0),206). orientation(lb1gt_1,151,206). coords(lb1gt_1,121,158,206).  visible(lb1gt_1,206). happensAt(walking(lb1gt_1),206).
orientation(lb1gt_0,136,207). coords(lb1gt_0,170,29,207).  visible(lb1gt_0,207). happensAt(walking(lb1gt_0),207). orientation(lb1gt_1,151,207). coords(lb1gt_1,121,158,207).  visible(lb1gt_1,207). happensAt(walking(lb1gt_1),207).
orientation(lb1gt_0,136,208). coords(lb1gt_0,171,29,208).  visible(lb1gt_0,208). happensAt(walking(lb1gt_0),208). orientation(lb1gt_1,151,208). coords(lb1gt_1,121,158,208).  visible(lb1gt_1,208). happensAt(walking(lb1gt_1),208).
orientation(lb1gt_0,136,209). coords(lb1gt_0,171,29,209).  visible(lb1gt_0,209). happensAt(walking(lb1gt_0),209). orientation(lb1gt_1,151,209). coords(lb1gt_1,121,159,209).  visible(lb1gt_1,209). happensAt(walking(lb1gt_1),209).
orientation(lb1gt_0,136,210). coords(lb1gt_0,171,29,210).  visible(lb1gt_0,210). happensAt(walking(lb1gt_0),210). orientation(lb1gt_1,151,210). coords(lb1gt_1,122,159,210).  visible(lb1gt_1,210). happensAt(walking(lb1gt_1),210).
orientation(lb1gt_0,136,211). coords(lb1gt_0,171,29,211).  visible(lb1gt_0,211). happensAt(walking(lb1gt_0),211). orientation(lb1gt_1,151,211). coords(lb1gt_1,122,159,211).  visible(lb1gt_1,211). happensAt(walking(lb1gt_1),211).
orientation(lb1gt_0,136,212). coords(lb1gt_0,171,29,212).  visible(lb1gt_0,212). happensAt(walking(lb1gt_0),212). orientation(lb1gt_1,151,212). coords(lb1gt_1,123,159,212).  visible(lb1gt_1,212). happensAt(walking(lb1gt_1),212).
orientation(lb1gt_0,136,213). coords(lb1gt_0,171,28,213).  visible(lb1gt_0,213). happensAt(walking(lb1gt_0),213). orientation(lb1gt_1,151,213). coords(lb1gt_1,124,159,213).  visible(lb1gt_1,213). happensAt(walking(lb1gt_1),213).
orientation(lb1gt_0,136,214). coords(lb1gt_0,172,28,214).  visible(lb1gt_0,214). happensAt(walking(lb1gt_0),214). orientation(lb1gt_1,151,214). coords(lb1gt_1,126,159,214).  visible(lb1gt_1,214). happensAt(walking(lb1gt_1),214).
orientation(lb1gt_0,136,215). coords(lb1gt_0,172,28,215).  visible(lb1gt_0,215). happensAt(walking(lb1gt_0),215). orientation(lb1gt_1,151,215). coords(lb1gt_1,127,159,215).  visible(lb1gt_1,215). happensAt(walking(lb1gt_1),215).
orientation(lb1gt_0,136,216). coords(lb1gt_0,173,28,216).  visible(lb1gt_0,216). happensAt(walking(lb1gt_0),216). orientation(lb1gt_1,151,216). coords(lb1gt_1,129,159,216).  visible(lb1gt_1,216). happensAt(walking(lb1gt_1),216).
#show holdsAt/2.