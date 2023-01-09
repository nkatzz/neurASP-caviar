
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

	orientation(fra1gt_6,101,198). coords(fra1gt_6,280,8,198).  visible(fra1gt_6,198). happensAt(walking(fra1gt_6),198). orientation(fra1gt_7,127,198). coords(fra1gt_7,168,23,198).  visible(fra1gt_7,198). happensAt(walking(fra1gt_7),198).
orientation(fra1gt_6,101,199). coords(fra1gt_6,280,8,199).  visible(fra1gt_6,199). happensAt(walking(fra1gt_6),199). orientation(fra1gt_7,125,199). coords(fra1gt_7,167,24,199).  visible(fra1gt_7,199). happensAt(walking(fra1gt_7),199).
orientation(fra1gt_6,101,200). coords(fra1gt_6,280,8,200).  visible(fra1gt_6,200). happensAt(walking(fra1gt_6),200). orientation(fra1gt_7,127,200). coords(fra1gt_7,167,26,200).  visible(fra1gt_7,200). happensAt(walking(fra1gt_7),200).
orientation(fra1gt_6,101,201). coords(fra1gt_6,279,8,201).  visible(fra1gt_6,201). happensAt(walking(fra1gt_6),201). orientation(fra1gt_7,127,201). coords(fra1gt_7,166,26,201).  visible(fra1gt_7,201). happensAt(walking(fra1gt_7),201).
orientation(fra1gt_6,101,202). coords(fra1gt_6,279,8,202).  visible(fra1gt_6,202). happensAt(walking(fra1gt_6),202). orientation(fra1gt_7,127,202). coords(fra1gt_7,166,26,202).  visible(fra1gt_7,202). happensAt(walking(fra1gt_7),202).
orientation(fra1gt_6,101,203). coords(fra1gt_6,278,8,203).  visible(fra1gt_6,203). happensAt(walking(fra1gt_6),203). orientation(fra1gt_7,127,203). coords(fra1gt_7,166,26,203).  visible(fra1gt_7,203). happensAt(walking(fra1gt_7),203).
orientation(fra1gt_6,101,204). coords(fra1gt_6,278,8,204).  visible(fra1gt_6,204). happensAt(walking(fra1gt_6),204). orientation(fra1gt_7,127,204). coords(fra1gt_7,166,27,204).  visible(fra1gt_7,204). happensAt(walking(fra1gt_7),204).
orientation(fra1gt_6,101,205). coords(fra1gt_6,278,10,205).  visible(fra1gt_6,205). happensAt(walking(fra1gt_6),205). orientation(fra1gt_7,127,205). coords(fra1gt_7,166,27,205).  visible(fra1gt_7,205). happensAt(walking(fra1gt_7),205).
orientation(fra1gt_6,101,206). coords(fra1gt_6,277,11,206).  visible(fra1gt_6,206). happensAt(walking(fra1gt_6),206). orientation(fra1gt_7,127,206). coords(fra1gt_7,165,28,206).  visible(fra1gt_7,206). happensAt(walking(fra1gt_7),206).
orientation(fra1gt_6,101,207). coords(fra1gt_6,277,12,207).  visible(fra1gt_6,207). happensAt(walking(fra1gt_6),207). orientation(fra1gt_7,127,207). coords(fra1gt_7,164,28,207).  visible(fra1gt_7,207). happensAt(walking(fra1gt_7),207).
orientation(fra1gt_6,101,208). coords(fra1gt_6,276,12,208).  visible(fra1gt_6,208). happensAt(walking(fra1gt_6),208). orientation(fra1gt_7,127,208). coords(fra1gt_7,163,28,208).  visible(fra1gt_7,208). happensAt(walking(fra1gt_7),208).
orientation(fra1gt_6,101,209). coords(fra1gt_6,276,13,209).  visible(fra1gt_6,209). happensAt(walking(fra1gt_6),209). orientation(fra1gt_7,127,209). coords(fra1gt_7,163,28,209).  visible(fra1gt_7,209). happensAt(walking(fra1gt_7),209).
orientation(fra1gt_6,101,210). coords(fra1gt_6,276,13,210).  visible(fra1gt_6,210). happensAt(walking(fra1gt_6),210). orientation(fra1gt_7,127,210). coords(fra1gt_7,163,28,210).  visible(fra1gt_7,210). happensAt(walking(fra1gt_7),210).
orientation(fra1gt_6,101,211). coords(fra1gt_6,275,14,211).  visible(fra1gt_6,211). happensAt(walking(fra1gt_6),211). orientation(fra1gt_7,127,211). coords(fra1gt_7,163,28,211).  visible(fra1gt_7,211). happensAt(walking(fra1gt_7),211).
orientation(fra1gt_6,101,212). coords(fra1gt_6,275,14,212).  visible(fra1gt_6,212). happensAt(walking(fra1gt_6),212). orientation(fra1gt_7,127,212). coords(fra1gt_7,163,29,212).  visible(fra1gt_7,212). happensAt(walking(fra1gt_7),212).
orientation(fra1gt_6,109,213). coords(fra1gt_6,275,14,213).  visible(fra1gt_6,213). happensAt(walking(fra1gt_6),213). orientation(fra1gt_7,127,213). coords(fra1gt_7,163,29,213).  visible(fra1gt_7,213). happensAt(walking(fra1gt_7),213).
orientation(fra1gt_6,109,214). coords(fra1gt_6,272,14,214).  visible(fra1gt_6,214). happensAt(walking(fra1gt_6),214). orientation(fra1gt_7,127,214). coords(fra1gt_7,162,30,214).  visible(fra1gt_7,214). happensAt(walking(fra1gt_7),214).
orientation(fra1gt_6,109,215). coords(fra1gt_6,272,14,215).  visible(fra1gt_6,215). happensAt(walking(fra1gt_6),215). orientation(fra1gt_7,127,215). coords(fra1gt_7,162,30,215).  visible(fra1gt_7,215). happensAt(walking(fra1gt_7),215).
orientation(fra1gt_6,109,216). coords(fra1gt_6,269,14,216).  visible(fra1gt_6,216). happensAt(walking(fra1gt_6),216). orientation(fra1gt_7,127,216). coords(fra1gt_7,162,31,216).  visible(fra1gt_7,216). happensAt(walking(fra1gt_7),216).
orientation(fra1gt_6,109,217). coords(fra1gt_6,268,14,217).  visible(fra1gt_6,217). happensAt(walking(fra1gt_6),217). orientation(fra1gt_7,127,217). coords(fra1gt_7,162,31,217).  visible(fra1gt_7,217). happensAt(walking(fra1gt_7),217).
orientation(fra1gt_6,109,218). coords(fra1gt_6,267,14,218).  visible(fra1gt_6,218). happensAt(walking(fra1gt_6),218). orientation(fra1gt_7,127,218). coords(fra1gt_7,161,31,218).  visible(fra1gt_7,218). happensAt(walking(fra1gt_7),218).
orientation(fra1gt_6,109,219). coords(fra1gt_6,264,15,219).  visible(fra1gt_6,219). happensAt(walking(fra1gt_6),219). orientation(fra1gt_7,127,219). coords(fra1gt_7,160,31,219).  visible(fra1gt_7,219). happensAt(walking(fra1gt_7),219).
orientation(fra1gt_6,109,220). coords(fra1gt_6,262,16,220).  visible(fra1gt_6,220). happensAt(walking(fra1gt_6),220). orientation(fra1gt_7,127,220). coords(fra1gt_7,159,31,220).  visible(fra1gt_7,220). happensAt(walking(fra1gt_7),220).
orientation(fra1gt_6,109,221). coords(fra1gt_6,262,18,221).  visible(fra1gt_6,221). happensAt(walking(fra1gt_6),221). orientation(fra1gt_7,127,221). coords(fra1gt_7,159,31,221).  visible(fra1gt_7,221). happensAt(walking(fra1gt_7),221).
orientation(fra1gt_6,109,222). coords(fra1gt_6,260,19,222).  visible(fra1gt_6,222). happensAt(walking(fra1gt_6),222). orientation(fra1gt_7,127,222). coords(fra1gt_7,159,32,222).  visible(fra1gt_7,222). happensAt(walking(fra1gt_7),222).
#show holdsAt/2.