
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

	orientation(fcgt_1,32,196). coords(fcgt_1,250,273,196).  visible(fcgt_1,196). happensAt(walking(fcgt_1),196). orientation(fcgt_3,159,196). coords(fcgt_3,103,163,196).  visible(fcgt_3,196). happensAt(inactive(fcgt_3),196).
orientation(fcgt_1,32,197). coords(fcgt_1,249,271,197).  visible(fcgt_1,197). happensAt(walking(fcgt_1),197). orientation(fcgt_3,159,197). coords(fcgt_3,103,163,197).  visible(fcgt_3,197). happensAt(inactive(fcgt_3),197).
orientation(fcgt_1,28,198). coords(fcgt_1,248,270,198).  visible(fcgt_1,198). happensAt(walking(fcgt_1),198). orientation(fcgt_3,159,198). coords(fcgt_3,103,163,198).  visible(fcgt_3,198). happensAt(inactive(fcgt_3),198).
orientation(fcgt_1,23,199). coords(fcgt_1,248,269,199).  visible(fcgt_1,199). happensAt(walking(fcgt_1),199). orientation(fcgt_3,159,199). coords(fcgt_3,103,163,199).  visible(fcgt_3,199). happensAt(inactive(fcgt_3),199).
orientation(fcgt_1,23,200). coords(fcgt_1,246,268,200).  visible(fcgt_1,200). happensAt(walking(fcgt_1),200). orientation(fcgt_3,159,200). coords(fcgt_3,103,163,200).  visible(fcgt_3,200). happensAt(inactive(fcgt_3),200).
orientation(fcgt_1,11,201). coords(fcgt_1,244,266,201).  visible(fcgt_1,201). happensAt(walking(fcgt_1),201). orientation(fcgt_3,159,201). coords(fcgt_3,102,163,201).  visible(fcgt_3,201). happensAt(inactive(fcgt_3),201).
orientation(fcgt_1,7,202). coords(fcgt_1,241,264,202).  visible(fcgt_1,202). happensAt(walking(fcgt_1),202). orientation(fcgt_3,159,202). coords(fcgt_3,102,163,202).  visible(fcgt_3,202). happensAt(walking(fcgt_3),202).
orientation(fcgt_1,7,203). coords(fcgt_1,238,262,203).  visible(fcgt_1,203). happensAt(walking(fcgt_1),203). orientation(fcgt_3,159,203). coords(fcgt_3,101,163,203).  visible(fcgt_3,203). happensAt(walking(fcgt_3),203).
orientation(fcgt_1,2,204). coords(fcgt_1,236,261,204).  visible(fcgt_1,204). happensAt(walking(fcgt_1),204). orientation(fcgt_3,159,204). coords(fcgt_3,101,164,204).  visible(fcgt_3,204). happensAt(walking(fcgt_3),204).
orientation(fcgt_1,2,205). coords(fcgt_1,233,258,205).  visible(fcgt_1,205). happensAt(walking(fcgt_1),205). orientation(fcgt_3,159,205). coords(fcgt_3,101,164,205).  visible(fcgt_3,205). happensAt(walking(fcgt_3),205).
orientation(fcgt_1,2,206). coords(fcgt_1,231,255,206).  visible(fcgt_1,206). happensAt(walking(fcgt_1),206). orientation(fcgt_3,159,206). coords(fcgt_3,101,165,206).  visible(fcgt_3,206). happensAt(walking(fcgt_3),206).
orientation(fcgt_1,2,207). coords(fcgt_1,229,249,207).  visible(fcgt_1,207). happensAt(walking(fcgt_1),207). orientation(fcgt_3,159,207). coords(fcgt_3,101,165,207).  visible(fcgt_3,207). happensAt(walking(fcgt_3),207).
orientation(fcgt_1,2,208). coords(fcgt_1,229,244,208).  visible(fcgt_1,208). happensAt(walking(fcgt_1),208). orientation(fcgt_3,153,208). coords(fcgt_3,100,166,208).  visible(fcgt_3,208). happensAt(walking(fcgt_3),208).
orientation(fcgt_1,2,209). coords(fcgt_1,228,242,209).  visible(fcgt_1,209). happensAt(walking(fcgt_1),209). orientation(fcgt_3,153,209). coords(fcgt_3,100,166,209).  visible(fcgt_3,209). happensAt(walking(fcgt_3),209).
orientation(fcgt_1,2,210). coords(fcgt_1,227,240,210).  visible(fcgt_1,210). happensAt(walking(fcgt_1),210). orientation(fcgt_3,149,210). coords(fcgt_3,100,166,210).  visible(fcgt_3,210). happensAt(walking(fcgt_3),210).
orientation(fcgt_1,177,211). coords(fcgt_1,226,240,211).  visible(fcgt_1,211). happensAt(walking(fcgt_1),211). orientation(fcgt_3,149,211). coords(fcgt_3,100,167,211).  visible(fcgt_3,211). happensAt(walking(fcgt_3),211).
orientation(fcgt_1,177,212). coords(fcgt_1,224,238,212).  visible(fcgt_1,212). happensAt(walking(fcgt_1),212). orientation(fcgt_3,149,212). coords(fcgt_3,100,167,212).  visible(fcgt_3,212). happensAt(walking(fcgt_3),212).
orientation(fcgt_1,169,213). coords(fcgt_1,222,237,213).  visible(fcgt_1,213). happensAt(walking(fcgt_1),213). orientation(fcgt_3,149,213). coords(fcgt_3,100,168,213).  visible(fcgt_3,213). happensAt(walking(fcgt_3),213).
orientation(fcgt_1,161,214). coords(fcgt_1,219,235,214).  visible(fcgt_1,214). happensAt(walking(fcgt_1),214). orientation(fcgt_3,149,214). coords(fcgt_3,100,168,214).  visible(fcgt_3,214). happensAt(walking(fcgt_3),214).
orientation(fcgt_1,161,215). coords(fcgt_1,215,232,215).  visible(fcgt_1,215). happensAt(walking(fcgt_1),215). orientation(fcgt_3,149,215). coords(fcgt_3,100,168,215).  visible(fcgt_3,215). happensAt(walking(fcgt_3),215).
orientation(fcgt_1,161,216). coords(fcgt_1,214,227,216).  visible(fcgt_1,216). happensAt(walking(fcgt_1),216). orientation(fcgt_3,149,216). coords(fcgt_3,100,168,216).  visible(fcgt_3,216). happensAt(walking(fcgt_3),216).
orientation(fcgt_1,171,217). coords(fcgt_1,213,222,217).  visible(fcgt_1,217). happensAt(walking(fcgt_1),217). orientation(fcgt_3,149,217). coords(fcgt_3,100,169,217).  visible(fcgt_3,217). happensAt(walking(fcgt_3),217).
orientation(fcgt_1,171,218). coords(fcgt_1,212,219,218).  visible(fcgt_1,218). happensAt(walking(fcgt_1),218). orientation(fcgt_3,149,218). coords(fcgt_3,100,169,218).  visible(fcgt_3,218). happensAt(walking(fcgt_3),218).
orientation(fcgt_1,171,219). coords(fcgt_1,211,215,219).  visible(fcgt_1,219). happensAt(walking(fcgt_1),219). orientation(fcgt_3,149,219). coords(fcgt_3,100,169,219).  visible(fcgt_3,219). happensAt(walking(fcgt_3),219).
orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220).
#show holdsAt/2.