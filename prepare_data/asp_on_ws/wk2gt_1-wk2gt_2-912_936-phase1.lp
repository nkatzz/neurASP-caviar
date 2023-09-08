
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

	orientation(wk2gt_1,127,912). coords(wk2gt_1,115,51,912).  visible(wk2gt_1,912). happensAt(inactive(wk2gt_1),912). orientation(wk2gt_2,142,912). coords(wk2gt_2,99,63,912).  visible(wk2gt_2,912). happensAt(inactive(wk2gt_2),912).
orientation(wk2gt_1,127,913). coords(wk2gt_1,115,51,913).  visible(wk2gt_1,913). happensAt(inactive(wk2gt_1),913). orientation(wk2gt_2,142,913). coords(wk2gt_2,99,63,913).  visible(wk2gt_2,913). happensAt(inactive(wk2gt_2),913).
orientation(wk2gt_1,127,914). coords(wk2gt_1,115,51,914).  visible(wk2gt_1,914). happensAt(inactive(wk2gt_1),914). orientation(wk2gt_2,142,914). coords(wk2gt_2,99,63,914).  visible(wk2gt_2,914). happensAt(inactive(wk2gt_2),914).
orientation(wk2gt_1,127,915). coords(wk2gt_1,115,51,915).  visible(wk2gt_1,915). happensAt(inactive(wk2gt_1),915). orientation(wk2gt_2,142,915). coords(wk2gt_2,99,63,915).  visible(wk2gt_2,915). happensAt(inactive(wk2gt_2),915).
orientation(wk2gt_1,127,916). coords(wk2gt_1,116,51,916).  visible(wk2gt_1,916). happensAt(inactive(wk2gt_1),916). orientation(wk2gt_2,142,916). coords(wk2gt_2,99,63,916).  visible(wk2gt_2,916). happensAt(inactive(wk2gt_2),916).
orientation(wk2gt_1,132,917). coords(wk2gt_1,116,51,917).  visible(wk2gt_1,917). happensAt(inactive(wk2gt_1),917). orientation(wk2gt_2,142,917). coords(wk2gt_2,99,63,917).  visible(wk2gt_2,917). happensAt(inactive(wk2gt_2),917).
orientation(wk2gt_1,132,918). coords(wk2gt_1,117,51,918).  visible(wk2gt_1,918). happensAt(inactive(wk2gt_1),918). orientation(wk2gt_2,142,918). coords(wk2gt_2,99,63,918).  visible(wk2gt_2,918). happensAt(inactive(wk2gt_2),918).
orientation(wk2gt_1,132,919). coords(wk2gt_1,117,51,919).  visible(wk2gt_1,919). happensAt(inactive(wk2gt_1),919). orientation(wk2gt_2,142,919). coords(wk2gt_2,99,63,919).  visible(wk2gt_2,919). happensAt(inactive(wk2gt_2),919).
orientation(wk2gt_1,133,920). coords(wk2gt_1,118,51,920).  visible(wk2gt_1,920). happensAt(inactive(wk2gt_1),920). orientation(wk2gt_2,142,920). coords(wk2gt_2,99,63,920).  visible(wk2gt_2,920). happensAt(inactive(wk2gt_2),920).
orientation(wk2gt_1,133,921). coords(wk2gt_1,118,51,921).  visible(wk2gt_1,921). happensAt(walking(wk2gt_1),921). orientation(wk2gt_2,142,921). coords(wk2gt_2,99,63,921).  visible(wk2gt_2,921). happensAt(inactive(wk2gt_2),921).
orientation(wk2gt_1,133,922). coords(wk2gt_1,119,50,922).  visible(wk2gt_1,922). happensAt(walking(wk2gt_1),922). orientation(wk2gt_2,142,922). coords(wk2gt_2,99,63,922).  visible(wk2gt_2,922). happensAt(inactive(wk2gt_2),922).
orientation(wk2gt_1,133,923). coords(wk2gt_1,119,50,923).  visible(wk2gt_1,923). happensAt(walking(wk2gt_1),923). orientation(wk2gt_2,142,923). coords(wk2gt_2,99,63,923).  visible(wk2gt_2,923). happensAt(inactive(wk2gt_2),923).
orientation(wk2gt_1,133,924). coords(wk2gt_1,120,50,924).  visible(wk2gt_1,924). happensAt(walking(wk2gt_1),924). orientation(wk2gt_2,142,924). coords(wk2gt_2,99,63,924).  visible(wk2gt_2,924). happensAt(inactive(wk2gt_2),924).
orientation(wk2gt_1,133,925). coords(wk2gt_1,120,51,925).  visible(wk2gt_1,925). happensAt(walking(wk2gt_1),925). orientation(wk2gt_2,142,925). coords(wk2gt_2,99,63,925).  visible(wk2gt_2,925). happensAt(inactive(wk2gt_2),925).
orientation(wk2gt_1,133,926). coords(wk2gt_1,120,51,926).  visible(wk2gt_1,926). happensAt(walking(wk2gt_1),926). orientation(wk2gt_2,142,926). coords(wk2gt_2,98,63,926).  visible(wk2gt_2,926). happensAt(inactive(wk2gt_2),926).
orientation(wk2gt_1,133,927). coords(wk2gt_1,120,51,927).  visible(wk2gt_1,927). happensAt(walking(wk2gt_1),927). orientation(wk2gt_2,142,927). coords(wk2gt_2,98,63,927).  visible(wk2gt_2,927). happensAt(inactive(wk2gt_2),927).
orientation(wk2gt_1,133,928). coords(wk2gt_1,120,51,928).  visible(wk2gt_1,928). happensAt(walking(wk2gt_1),928). orientation(wk2gt_2,142,928). coords(wk2gt_2,98,63,928).  visible(wk2gt_2,928). happensAt(inactive(wk2gt_2),928).
orientation(wk2gt_1,133,929). coords(wk2gt_1,120,51,929).  visible(wk2gt_1,929). happensAt(walking(wk2gt_1),929). orientation(wk2gt_2,142,929). coords(wk2gt_2,98,63,929).  visible(wk2gt_2,929). happensAt(inactive(wk2gt_2),929).
orientation(wk2gt_1,133,930). coords(wk2gt_1,120,50,930).  visible(wk2gt_1,930). happensAt(walking(wk2gt_1),930). orientation(wk2gt_2,142,930). coords(wk2gt_2,98,63,930).  visible(wk2gt_2,930). happensAt(inactive(wk2gt_2),930).
orientation(wk2gt_1,133,931). coords(wk2gt_1,120,49,931).  visible(wk2gt_1,931). happensAt(walking(wk2gt_1),931). orientation(wk2gt_2,142,931). coords(wk2gt_2,98,63,931).  visible(wk2gt_2,931). happensAt(inactive(wk2gt_2),931).
orientation(wk2gt_1,133,932). coords(wk2gt_1,120,49,932).  visible(wk2gt_1,932). happensAt(walking(wk2gt_1),932). orientation(wk2gt_2,142,932). coords(wk2gt_2,98,63,932).  visible(wk2gt_2,932). happensAt(inactive(wk2gt_2),932).
orientation(wk2gt_1,133,933). coords(wk2gt_1,120,49,933).  visible(wk2gt_1,933). happensAt(walking(wk2gt_1),933). orientation(wk2gt_2,142,933). coords(wk2gt_2,98,63,933).  visible(wk2gt_2,933). happensAt(inactive(wk2gt_2),933).
orientation(wk2gt_1,133,934). coords(wk2gt_1,120,48,934).  visible(wk2gt_1,934). happensAt(walking(wk2gt_1),934). orientation(wk2gt_2,142,934). coords(wk2gt_2,98,64,934).  visible(wk2gt_2,934). happensAt(inactive(wk2gt_2),934).
orientation(wk2gt_1,133,935). coords(wk2gt_1,120,48,935).  visible(wk2gt_1,935). happensAt(walking(wk2gt_1),935). orientation(wk2gt_2,142,935). coords(wk2gt_2,98,64,935).  visible(wk2gt_2,935). happensAt(inactive(wk2gt_2),935).
orientation(wk2gt_1,133,936). coords(wk2gt_1,121,48,936).  visible(wk2gt_1,936). happensAt(walking(wk2gt_1),936). orientation(wk2gt_2,138,936). coords(wk2gt_2,99,65,936).  visible(wk2gt_2,936). happensAt(active(wk2gt_2),936).
#show holdsAt/2.