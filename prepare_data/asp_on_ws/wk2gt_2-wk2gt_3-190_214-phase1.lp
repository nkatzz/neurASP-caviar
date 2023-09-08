
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

	orientation(wk2gt_2,141,190). coords(wk2gt_2,98,62,190).  visible(wk2gt_2,190). happensAt(inactive(wk2gt_2),190). orientation(wk2gt_3,157,190). coords(wk2gt_3,237,216,190).  visible(wk2gt_3,190). happensAt(walking(wk2gt_3),190).
orientation(wk2gt_2,141,191). coords(wk2gt_2,98,62,191).  visible(wk2gt_2,191). happensAt(inactive(wk2gt_2),191). orientation(wk2gt_3,162,191). coords(wk2gt_3,239,220,191).  visible(wk2gt_3,191). happensAt(walking(wk2gt_3),191).
orientation(wk2gt_2,141,192). coords(wk2gt_2,98,62,192).  visible(wk2gt_2,192). happensAt(inactive(wk2gt_2),192). orientation(wk2gt_3,169,192). coords(wk2gt_3,242,224,192).  visible(wk2gt_3,192). happensAt(walking(wk2gt_3),192).
orientation(wk2gt_2,141,193). coords(wk2gt_2,98,62,193).  visible(wk2gt_2,193). happensAt(inactive(wk2gt_2),193). orientation(wk2gt_3,174,193). coords(wk2gt_3,245,226,193).  visible(wk2gt_3,193). happensAt(walking(wk2gt_3),193).
orientation(wk2gt_2,141,194). coords(wk2gt_2,98,62,194).  visible(wk2gt_2,194). happensAt(inactive(wk2gt_2),194). orientation(wk2gt_3,170,194). coords(wk2gt_3,248,228,194).  visible(wk2gt_3,194). happensAt(walking(wk2gt_3),194).
orientation(wk2gt_2,141,195). coords(wk2gt_2,98,62,195).  visible(wk2gt_2,195). happensAt(inactive(wk2gt_2),195). orientation(wk2gt_3,9,195). coords(wk2gt_3,250,228,195).  visible(wk2gt_3,195). happensAt(walking(wk2gt_3),195).
orientation(wk2gt_2,141,196). coords(wk2gt_2,98,62,196).  visible(wk2gt_2,196). happensAt(inactive(wk2gt_2),196). orientation(wk2gt_3,29,196). coords(wk2gt_3,250,230,196).  visible(wk2gt_3,196). happensAt(walking(wk2gt_3),196).
orientation(wk2gt_2,141,197). coords(wk2gt_2,98,62,197).  visible(wk2gt_2,197). happensAt(inactive(wk2gt_2),197). orientation(wk2gt_3,40,197). coords(wk2gt_3,251,231,197).  visible(wk2gt_3,197). happensAt(walking(wk2gt_3),197).
orientation(wk2gt_2,141,198). coords(wk2gt_2,98,62,198).  visible(wk2gt_2,198). happensAt(inactive(wk2gt_2),198). orientation(wk2gt_3,37,198). coords(wk2gt_3,253,233,198).  visible(wk2gt_3,198). happensAt(walking(wk2gt_3),198).
orientation(wk2gt_2,141,199). coords(wk2gt_2,98,62,199).  visible(wk2gt_2,199). happensAt(inactive(wk2gt_2),199). orientation(wk2gt_3,38,199). coords(wk2gt_3,255,237,199).  visible(wk2gt_3,199). happensAt(walking(wk2gt_3),199).
orientation(wk2gt_2,141,200). coords(wk2gt_2,98,62,200).  visible(wk2gt_2,200). happensAt(inactive(wk2gt_2),200). orientation(wk2gt_3,38,200). coords(wk2gt_3,258,241,200).  visible(wk2gt_3,200). happensAt(walking(wk2gt_3),200).
orientation(wk2gt_2,141,201). coords(wk2gt_2,98,62,201).  visible(wk2gt_2,201). happensAt(inactive(wk2gt_2),201). orientation(wk2gt_3,38,201). coords(wk2gt_3,262,245,201).  visible(wk2gt_3,201). happensAt(walking(wk2gt_3),201).
orientation(wk2gt_2,141,202). coords(wk2gt_2,98,62,202).  visible(wk2gt_2,202). happensAt(inactive(wk2gt_2),202). orientation(wk2gt_3,38,202). coords(wk2gt_3,265,247,202).  visible(wk2gt_3,202). happensAt(walking(wk2gt_3),202).
orientation(wk2gt_2,141,203). coords(wk2gt_2,98,62,203).  visible(wk2gt_2,203). happensAt(inactive(wk2gt_2),203). orientation(wk2gt_3,38,203). coords(wk2gt_3,269,249,203).  visible(wk2gt_3,203). happensAt(walking(wk2gt_3),203).
orientation(wk2gt_2,141,204). coords(wk2gt_2,98,62,204).  visible(wk2gt_2,204). happensAt(inactive(wk2gt_2),204). orientation(wk2gt_3,38,204). coords(wk2gt_3,272,252,204).  visible(wk2gt_3,204). happensAt(walking(wk2gt_3),204).
orientation(wk2gt_2,141,205). coords(wk2gt_2,98,62,205).  visible(wk2gt_2,205). happensAt(inactive(wk2gt_2),205). orientation(wk2gt_3,38,205). coords(wk2gt_3,275,254,205).  visible(wk2gt_3,205). happensAt(walking(wk2gt_3),205).
orientation(wk2gt_2,141,206). coords(wk2gt_2,98,62,206).  visible(wk2gt_2,206). happensAt(inactive(wk2gt_2),206). orientation(wk2gt_3,38,206). coords(wk2gt_3,277,257,206).  visible(wk2gt_3,206). happensAt(walking(wk2gt_3),206).
orientation(wk2gt_2,141,207). coords(wk2gt_2,98,62,207).  visible(wk2gt_2,207). happensAt(inactive(wk2gt_2),207). orientation(wk2gt_3,38,207). coords(wk2gt_3,279,258,207).  visible(wk2gt_3,207). happensAt(walking(wk2gt_3),207).
orientation(wk2gt_2,141,208). coords(wk2gt_2,98,62,208).  visible(wk2gt_2,208). happensAt(inactive(wk2gt_2),208). orientation(wk2gt_3,38,208). coords(wk2gt_3,280,260,208).  visible(wk2gt_3,208). happensAt(walking(wk2gt_3),208).
orientation(wk2gt_2,141,209). coords(wk2gt_2,98,62,209).  visible(wk2gt_2,209). happensAt(inactive(wk2gt_2),209). orientation(wk2gt_3,38,209). coords(wk2gt_3,281,262,209).  visible(wk2gt_3,209). happensAt(walking(wk2gt_3),209).
orientation(wk2gt_2,141,210). coords(wk2gt_2,98,62,210).  visible(wk2gt_2,210). happensAt(inactive(wk2gt_2),210). orientation(wk2gt_3,38,210). coords(wk2gt_3,282,265,210).  visible(wk2gt_3,210). happensAt(walking(wk2gt_3),210).
orientation(wk2gt_2,141,211). coords(wk2gt_2,98,62,211).  visible(wk2gt_2,211). happensAt(inactive(wk2gt_2),211). orientation(wk2gt_3,46,211). coords(wk2gt_3,282,267,211).  visible(wk2gt_3,211). happensAt(walking(wk2gt_3),211).
orientation(wk2gt_2,141,212). coords(wk2gt_2,98,62,212).  visible(wk2gt_2,212). happensAt(inactive(wk2gt_2),212). orientation(wk2gt_3,48,212). coords(wk2gt_3,284,270,212).  visible(wk2gt_3,212). happensAt(walking(wk2gt_3),212).
orientation(wk2gt_2,141,213). coords(wk2gt_2,98,62,213).  visible(wk2gt_2,213). happensAt(inactive(wk2gt_2),213). orientation(wk2gt_3,48,213). coords(wk2gt_3,287,272,213).  visible(wk2gt_3,213). happensAt(walking(wk2gt_3),213).
orientation(wk2gt_2,141,214). coords(wk2gt_2,98,62,214).  visible(wk2gt_2,214). happensAt(inactive(wk2gt_2),214). orientation(wk2gt_3,48,214). coords(wk2gt_3,288,273,214).  visible(wk2gt_3,214). happensAt(walking(wk2gt_3),214).
#show holdsAt/2.