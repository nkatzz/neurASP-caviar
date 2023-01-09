
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

	orientation(mws1gt_3,162,202). coords(mws1gt_3,139,155,202).  visible(mws1gt_3,202). happensAt(walking(mws1gt_3),202). orientation(mws1gt_5,161,202). coords(mws1gt_5,47,103,202).  visible(mws1gt_5,202). happensAt(walking(mws1gt_5),202).
orientation(mws1gt_3,162,203). coords(mws1gt_3,139,155,203).  visible(mws1gt_3,203). happensAt(walking(mws1gt_3),203). orientation(mws1gt_5,161,203). coords(mws1gt_5,46,104,203).  visible(mws1gt_5,203). happensAt(walking(mws1gt_5),203).
orientation(mws1gt_3,162,204). coords(mws1gt_3,139,155,204).  visible(mws1gt_3,204). happensAt(walking(mws1gt_3),204). orientation(mws1gt_5,161,204). coords(mws1gt_5,45,105,204).  visible(mws1gt_5,204). happensAt(walking(mws1gt_5),204).
orientation(mws1gt_3,162,205). coords(mws1gt_3,139,155,205).  visible(mws1gt_3,205). happensAt(walking(mws1gt_3),205). orientation(mws1gt_5,161,205). coords(mws1gt_5,45,107,205).  visible(mws1gt_5,205). happensAt(walking(mws1gt_5),205).
orientation(mws1gt_3,162,206). coords(mws1gt_3,138,155,206).  visible(mws1gt_3,206). happensAt(walking(mws1gt_3),206). orientation(mws1gt_5,161,206). coords(mws1gt_5,45,107,206).  visible(mws1gt_5,206). happensAt(walking(mws1gt_5),206).
orientation(mws1gt_3,162,207). coords(mws1gt_3,138,155,207).  visible(mws1gt_3,207). happensAt(walking(mws1gt_3),207). orientation(mws1gt_5,161,207). coords(mws1gt_5,45,107,207).  visible(mws1gt_5,207). happensAt(walking(mws1gt_5),207).
orientation(mws1gt_3,162,208). coords(mws1gt_3,137,155,208).  visible(mws1gt_3,208). happensAt(walking(mws1gt_3),208). orientation(mws1gt_5,161,208). coords(mws1gt_5,45,108,208).  visible(mws1gt_5,208). happensAt(walking(mws1gt_5),208).
orientation(mws1gt_3,162,209). coords(mws1gt_3,135,155,209).  visible(mws1gt_3,209). happensAt(walking(mws1gt_3),209). orientation(mws1gt_5,161,209). coords(mws1gt_5,45,109,209).  visible(mws1gt_5,209). happensAt(walking(mws1gt_5),209).
orientation(mws1gt_3,162,210). coords(mws1gt_3,134,155,210).  visible(mws1gt_3,210). happensAt(walking(mws1gt_3),210). orientation(mws1gt_5,161,210). coords(mws1gt_5,45,109,210).  visible(mws1gt_5,210). happensAt(walking(mws1gt_5),210).
orientation(mws1gt_3,162,211). coords(mws1gt_3,132,155,211).  visible(mws1gt_3,211). happensAt(walking(mws1gt_3),211). orientation(mws1gt_5,161,211). coords(mws1gt_5,45,109,211).  visible(mws1gt_5,211). happensAt(walking(mws1gt_5),211).
orientation(mws1gt_3,162,212). coords(mws1gt_3,130,155,212).  visible(mws1gt_3,212). happensAt(walking(mws1gt_3),212). orientation(mws1gt_5,161,212). coords(mws1gt_5,45,110,212).  visible(mws1gt_5,212). happensAt(walking(mws1gt_5),212).
orientation(mws1gt_3,162,213). coords(mws1gt_3,127,155,213).  visible(mws1gt_3,213). happensAt(walking(mws1gt_3),213). orientation(mws1gt_5,161,213). coords(mws1gt_5,45,111,213).  visible(mws1gt_5,213). happensAt(walking(mws1gt_5),213).
orientation(mws1gt_3,162,214). coords(mws1gt_3,126,154,214).  visible(mws1gt_3,214). happensAt(walking(mws1gt_3),214). orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). happensAt(walking(mws1gt_5),214).
orientation(mws1gt_3,162,215). coords(mws1gt_3,125,154,215).  visible(mws1gt_3,215). happensAt(walking(mws1gt_3),215). orientation(mws1gt_5,161,215). coords(mws1gt_5,46,116,215).  visible(mws1gt_5,215). happensAt(walking(mws1gt_5),215).
orientation(mws1gt_3,162,216). coords(mws1gt_3,125,154,216).  visible(mws1gt_3,216). happensAt(walking(mws1gt_3),216). orientation(mws1gt_5,161,216). coords(mws1gt_5,46,117,216).  visible(mws1gt_5,216). happensAt(walking(mws1gt_5),216).
orientation(mws1gt_3,162,217). coords(mws1gt_3,125,154,217).  visible(mws1gt_3,217). happensAt(walking(mws1gt_3),217). orientation(mws1gt_5,161,217). coords(mws1gt_5,46,117,217).  visible(mws1gt_5,217). happensAt(walking(mws1gt_5),217).
orientation(mws1gt_3,162,218). coords(mws1gt_3,124,153,218).  visible(mws1gt_3,218). happensAt(walking(mws1gt_3),218). orientation(mws1gt_5,161,218). coords(mws1gt_5,46,118,218).  visible(mws1gt_5,218). happensAt(walking(mws1gt_5),218).
orientation(mws1gt_3,162,219). coords(mws1gt_3,124,153,219).  visible(mws1gt_3,219). happensAt(walking(mws1gt_3),219). orientation(mws1gt_5,161,219). coords(mws1gt_5,46,118,219).  visible(mws1gt_5,219). happensAt(walking(mws1gt_5),219).
orientation(mws1gt_3,162,220). coords(mws1gt_3,123,153,220).  visible(mws1gt_3,220). happensAt(walking(mws1gt_3),220). orientation(mws1gt_5,161,220). coords(mws1gt_5,46,119,220).  visible(mws1gt_5,220). happensAt(walking(mws1gt_5),220).
orientation(mws1gt_3,162,221). coords(mws1gt_3,123,153,221).  visible(mws1gt_3,221). happensAt(walking(mws1gt_3),221). orientation(mws1gt_5,161,221). coords(mws1gt_5,46,120,221).  visible(mws1gt_5,221). happensAt(walking(mws1gt_5),221).
orientation(mws1gt_3,162,222). coords(mws1gt_3,123,153,222).  visible(mws1gt_3,222). happensAt(walking(mws1gt_3),222). orientation(mws1gt_5,161,222). coords(mws1gt_5,46,120,222).  visible(mws1gt_5,222). happensAt(walking(mws1gt_5),222).
orientation(mws1gt_3,162,223). coords(mws1gt_3,122,152,223).  visible(mws1gt_3,223). happensAt(walking(mws1gt_3),223). orientation(mws1gt_5,161,223). coords(mws1gt_5,46,120,223).  visible(mws1gt_5,223). happensAt(walking(mws1gt_5),223).
orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224). happensAt(walking(mws1gt_3),224). orientation(mws1gt_5,161,224). coords(mws1gt_5,46,120,224).  visible(mws1gt_5,224). happensAt(walking(mws1gt_5),224).
orientation(mws1gt_3,162,225). coords(mws1gt_3,120,151,225).  visible(mws1gt_3,225). happensAt(walking(mws1gt_3),225). orientation(mws1gt_5,161,225). coords(mws1gt_5,46,120,225).  visible(mws1gt_5,225). happensAt(walking(mws1gt_5),225).
orientation(mws1gt_3,162,226). coords(mws1gt_3,118,150,226).  visible(mws1gt_3,226). happensAt(walking(mws1gt_3),226). orientation(mws1gt_5,161,226). coords(mws1gt_5,45,120,226).  visible(mws1gt_5,226). happensAt(walking(mws1gt_5),226).
#show holdsAt/2.