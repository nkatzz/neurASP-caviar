
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

	orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192). happensAt(walking(mc1gt_1),192). orientation(mc1gt_3,146,192). coords(mc1gt_3,118,82,192).  visible(mc1gt_3,192). happensAt(walking(mc1gt_3),192).
orientation(mc1gt_1,140,193). coords(mc1gt_1,152,97,193).  visible(mc1gt_1,193). happensAt(walking(mc1gt_1),193). orientation(mc1gt_3,146,193). coords(mc1gt_3,118,82,193).  visible(mc1gt_3,193). happensAt(walking(mc1gt_3),193).
orientation(mc1gt_1,140,194). coords(mc1gt_1,152,97,194).  visible(mc1gt_1,194). happensAt(walking(mc1gt_1),194). orientation(mc1gt_3,146,194). coords(mc1gt_3,119,82,194).  visible(mc1gt_3,194). happensAt(walking(mc1gt_3),194).
orientation(mc1gt_1,140,195). coords(mc1gt_1,152,97,195).  visible(mc1gt_1,195). happensAt(walking(mc1gt_1),195). orientation(mc1gt_3,146,195). coords(mc1gt_3,120,82,195).  visible(mc1gt_3,195). happensAt(walking(mc1gt_3),195).
orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196). happensAt(walking(mc1gt_1),196). orientation(mc1gt_3,146,196). coords(mc1gt_3,120,82,196).  visible(mc1gt_3,196). happensAt(walking(mc1gt_3),196).
orientation(mc1gt_1,140,197). coords(mc1gt_1,153,100,197).  visible(mc1gt_1,197). happensAt(walking(mc1gt_1),197). orientation(mc1gt_3,146,197). coords(mc1gt_3,121,82,197).  visible(mc1gt_3,197). happensAt(walking(mc1gt_3),197).
orientation(mc1gt_1,139,198). coords(mc1gt_1,154,101,198).  visible(mc1gt_1,198). happensAt(walking(mc1gt_1),198). orientation(mc1gt_3,146,198). coords(mc1gt_3,122,82,198).  visible(mc1gt_3,198). happensAt(walking(mc1gt_3),198).
orientation(mc1gt_1,139,199). coords(mc1gt_1,155,103,199).  visible(mc1gt_1,199). happensAt(walking(mc1gt_1),199). orientation(mc1gt_3,146,199). coords(mc1gt_3,123,82,199).  visible(mc1gt_3,199). happensAt(walking(mc1gt_3),199).
orientation(mc1gt_1,139,200). coords(mc1gt_1,157,105,200).  visible(mc1gt_1,200). happensAt(walking(mc1gt_1),200). orientation(mc1gt_3,146,200). coords(mc1gt_3,126,83,200).  visible(mc1gt_3,200). happensAt(walking(mc1gt_3),200).
orientation(mc1gt_1,137,201). coords(mc1gt_1,159,106,201).  visible(mc1gt_1,201). happensAt(walking(mc1gt_1),201). orientation(mc1gt_3,146,201). coords(mc1gt_3,127,83,201).  visible(mc1gt_3,201). happensAt(walking(mc1gt_3),201).
orientation(mc1gt_1,137,202). coords(mc1gt_1,160,107,202).  visible(mc1gt_1,202). happensAt(walking(mc1gt_1),202). orientation(mc1gt_3,146,202). coords(mc1gt_3,129,83,202).  visible(mc1gt_3,202). happensAt(walking(mc1gt_3),202).
orientation(mc1gt_1,137,203). coords(mc1gt_1,161,108,203).  visible(mc1gt_1,203). happensAt(walking(mc1gt_1),203). orientation(mc1gt_3,146,203). coords(mc1gt_3,131,83,203).  visible(mc1gt_3,203). happensAt(walking(mc1gt_3),203).
orientation(mc1gt_1,137,204). coords(mc1gt_1,161,109,204).  visible(mc1gt_1,204). happensAt(walking(mc1gt_1),204). orientation(mc1gt_3,146,204). coords(mc1gt_3,133,84,204).  visible(mc1gt_3,204). happensAt(walking(mc1gt_3),204).
orientation(mc1gt_1,137,205). coords(mc1gt_1,162,109,205).  visible(mc1gt_1,205). happensAt(walking(mc1gt_1),205). orientation(mc1gt_3,146,205). coords(mc1gt_3,134,84,205).  visible(mc1gt_3,205). happensAt(walking(mc1gt_3),205).
orientation(mc1gt_1,140,206). coords(mc1gt_1,162,110,206).  visible(mc1gt_1,206). happensAt(walking(mc1gt_1),206). orientation(mc1gt_3,146,206). coords(mc1gt_3,136,84,206).  visible(mc1gt_3,206). happensAt(walking(mc1gt_3),206).
orientation(mc1gt_1,140,207). coords(mc1gt_1,162,110,207).  visible(mc1gt_1,207). happensAt(walking(mc1gt_1),207). orientation(mc1gt_3,146,207). coords(mc1gt_3,137,82,207).  visible(mc1gt_3,207). happensAt(walking(mc1gt_3),207).
orientation(mc1gt_1,140,208). coords(mc1gt_1,162,110,208).  visible(mc1gt_1,208). happensAt(walking(mc1gt_1),208). orientation(mc1gt_3,146,208). coords(mc1gt_3,138,82,208).  visible(mc1gt_3,208). happensAt(walking(mc1gt_3),208).
orientation(mc1gt_1,141,209). coords(mc1gt_1,163,111,209).  visible(mc1gt_1,209). happensAt(walking(mc1gt_1),209). orientation(mc1gt_3,146,209). coords(mc1gt_3,138,83,209).  visible(mc1gt_3,209). happensAt(walking(mc1gt_3),209).
orientation(mc1gt_1,141,210). coords(mc1gt_1,163,111,210).  visible(mc1gt_1,210). happensAt(walking(mc1gt_1),210). orientation(mc1gt_3,146,210). coords(mc1gt_3,139,83,210).  visible(mc1gt_3,210). happensAt(walking(mc1gt_3),210).
orientation(mc1gt_1,143,211). coords(mc1gt_1,163,111,211).  visible(mc1gt_1,211). happensAt(walking(mc1gt_1),211). orientation(mc1gt_3,146,211). coords(mc1gt_3,141,83,211).  visible(mc1gt_3,211). happensAt(walking(mc1gt_3),211).
orientation(mc1gt_1,142,212). coords(mc1gt_1,164,112,212).  visible(mc1gt_1,212). happensAt(walking(mc1gt_1),212). orientation(mc1gt_3,146,212). coords(mc1gt_3,142,83,212).  visible(mc1gt_3,212). happensAt(walking(mc1gt_3),212).
orientation(mc1gt_1,143,213). coords(mc1gt_1,167,112,213).  visible(mc1gt_1,213). happensAt(walking(mc1gt_1),213). orientation(mc1gt_3,146,213). coords(mc1gt_3,142,83,213).  visible(mc1gt_3,213). happensAt(walking(mc1gt_3),213).
orientation(mc1gt_1,143,214). coords(mc1gt_1,169,113,214).  visible(mc1gt_1,214). happensAt(walking(mc1gt_1),214). orientation(mc1gt_3,146,214). coords(mc1gt_3,143,83,214).  visible(mc1gt_3,214). happensAt(walking(mc1gt_3),214).
orientation(mc1gt_1,143,215). coords(mc1gt_1,171,114,215).  visible(mc1gt_1,215). happensAt(walking(mc1gt_1),215). orientation(mc1gt_3,146,215). coords(mc1gt_3,144,84,215).  visible(mc1gt_3,215). happensAt(walking(mc1gt_3),215).
orientation(mc1gt_1,142,216). coords(mc1gt_1,173,115,216).  visible(mc1gt_1,216). happensAt(walking(mc1gt_1),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
#show holdsAt/2.