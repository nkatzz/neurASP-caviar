
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

	orientation(lbbcgt_5,9,909). coords(lbbcgt_5,224,239,909).  visible(lbbcgt_5,909). happensAt(walking(lbbcgt_5),909). orientation(lbbcgt_6,142,909). coords(lbbcgt_6,118,74,909).  visible(lbbcgt_6,909). happensAt(walking(lbbcgt_6),909).
orientation(lbbcgt_5,9,910). coords(lbbcgt_5,223,238,910).  visible(lbbcgt_5,910). happensAt(walking(lbbcgt_5),910). orientation(lbbcgt_6,142,910). coords(lbbcgt_6,118,76,910).  visible(lbbcgt_6,910). happensAt(walking(lbbcgt_6),910).
orientation(lbbcgt_5,9,911). coords(lbbcgt_5,221,236,911).  visible(lbbcgt_5,911). happensAt(walking(lbbcgt_5),911). orientation(lbbcgt_6,142,911). coords(lbbcgt_6,117,75,911).  visible(lbbcgt_6,911). happensAt(walking(lbbcgt_6),911).
orientation(lbbcgt_5,9,912). coords(lbbcgt_5,220,235,912).  visible(lbbcgt_5,912). happensAt(walking(lbbcgt_5),912). orientation(lbbcgt_6,144,912). coords(lbbcgt_6,115,76,912).  visible(lbbcgt_6,912). happensAt(walking(lbbcgt_6),912).
orientation(lbbcgt_5,9,913). coords(lbbcgt_5,218,230,913).  visible(lbbcgt_5,913). happensAt(walking(lbbcgt_5),913). orientation(lbbcgt_6,144,913). coords(lbbcgt_6,113,76,913).  visible(lbbcgt_6,913). happensAt(walking(lbbcgt_6),913).
orientation(lbbcgt_5,9,914). coords(lbbcgt_5,217,227,914).  visible(lbbcgt_5,914). happensAt(walking(lbbcgt_5),914). orientation(lbbcgt_6,144,914). coords(lbbcgt_6,112,76,914).  visible(lbbcgt_6,914). happensAt(walking(lbbcgt_6),914).
orientation(lbbcgt_5,9,915). coords(lbbcgt_5,215,225,915).  visible(lbbcgt_5,915). happensAt(walking(lbbcgt_5),915). orientation(lbbcgt_6,144,915). coords(lbbcgt_6,110,77,915).  visible(lbbcgt_6,915). happensAt(walking(lbbcgt_6),915).
orientation(lbbcgt_5,9,916). coords(lbbcgt_5,214,222,916).  visible(lbbcgt_5,916). happensAt(walking(lbbcgt_5),916). orientation(lbbcgt_6,144,916). coords(lbbcgt_6,110,77,916).  visible(lbbcgt_6,916). happensAt(walking(lbbcgt_6),916).
orientation(lbbcgt_5,9,917). coords(lbbcgt_5,212,219,917).  visible(lbbcgt_5,917). happensAt(walking(lbbcgt_5),917). orientation(lbbcgt_6,144,917). coords(lbbcgt_6,109,79,917).  visible(lbbcgt_6,917). happensAt(walking(lbbcgt_6),917).
orientation(lbbcgt_5,9,918). coords(lbbcgt_5,212,217,918).  visible(lbbcgt_5,918). happensAt(walking(lbbcgt_5),918). orientation(lbbcgt_6,144,918). coords(lbbcgt_6,108,79,918).  visible(lbbcgt_6,918). happensAt(walking(lbbcgt_6),918).
orientation(lbbcgt_5,173,919). coords(lbbcgt_5,212,214,919).  visible(lbbcgt_5,919). happensAt(walking(lbbcgt_5),919). orientation(lbbcgt_6,144,919). coords(lbbcgt_6,107,80,919).  visible(lbbcgt_6,919). happensAt(walking(lbbcgt_6),919).
orientation(lbbcgt_5,167,920). coords(lbbcgt_5,212,213,920).  visible(lbbcgt_5,920). happensAt(walking(lbbcgt_5),920). orientation(lbbcgt_6,144,920). coords(lbbcgt_6,107,81,920).  visible(lbbcgt_6,920). happensAt(walking(lbbcgt_6),920).
orientation(lbbcgt_5,167,921). coords(lbbcgt_5,212,213,921).  visible(lbbcgt_5,921). happensAt(walking(lbbcgt_5),921). orientation(lbbcgt_6,144,921). coords(lbbcgt_6,107,81,921).  visible(lbbcgt_6,921). happensAt(walking(lbbcgt_6),921).
orientation(lbbcgt_5,167,922). coords(lbbcgt_5,211,213,922).  visible(lbbcgt_5,922). happensAt(walking(lbbcgt_5),922). orientation(lbbcgt_6,144,922). coords(lbbcgt_6,106,81,922).  visible(lbbcgt_6,922). happensAt(walking(lbbcgt_6),922).
orientation(lbbcgt_5,167,923). coords(lbbcgt_5,210,212,923).  visible(lbbcgt_5,923). happensAt(walking(lbbcgt_5),923). orientation(lbbcgt_6,144,923). coords(lbbcgt_6,105,81,923).  visible(lbbcgt_6,923). happensAt(walking(lbbcgt_6),923).
orientation(lbbcgt_5,164,924). coords(lbbcgt_5,209,212,924).  visible(lbbcgt_5,924). happensAt(walking(lbbcgt_5),924). orientation(lbbcgt_6,144,924). coords(lbbcgt_6,105,81,924).  visible(lbbcgt_6,924). happensAt(walking(lbbcgt_6),924).
orientation(lbbcgt_5,164,925). coords(lbbcgt_5,208,210,925).  visible(lbbcgt_5,925). happensAt(walking(lbbcgt_5),925). orientation(lbbcgt_6,144,925). coords(lbbcgt_6,104,82,925).  visible(lbbcgt_6,925). happensAt(walking(lbbcgt_6),925).
orientation(lbbcgt_5,164,926). coords(lbbcgt_5,207,207,926).  visible(lbbcgt_5,926). happensAt(walking(lbbcgt_5),926). orientation(lbbcgt_6,144,926). coords(lbbcgt_6,103,82,926).  visible(lbbcgt_6,926). happensAt(walking(lbbcgt_6),926).
orientation(lbbcgt_5,164,927). coords(lbbcgt_5,206,204,927).  visible(lbbcgt_5,927). happensAt(walking(lbbcgt_5),927). orientation(lbbcgt_6,144,927). coords(lbbcgt_6,101,82,927).  visible(lbbcgt_6,927). happensAt(walking(lbbcgt_6),927).
orientation(lbbcgt_5,158,928). coords(lbbcgt_5,206,201,928).  visible(lbbcgt_5,928). happensAt(walking(lbbcgt_5),928). orientation(lbbcgt_6,144,928). coords(lbbcgt_6,100,82,928).  visible(lbbcgt_6,928). happensAt(walking(lbbcgt_6),928).
orientation(lbbcgt_5,158,929). coords(lbbcgt_5,205,198,929).  visible(lbbcgt_5,929). happensAt(walking(lbbcgt_5),929). orientation(lbbcgt_6,144,929). coords(lbbcgt_6,99,82,929).  visible(lbbcgt_6,929). happensAt(walking(lbbcgt_6),929).
orientation(lbbcgt_5,158,930). coords(lbbcgt_5,205,193,930).  visible(lbbcgt_5,930). happensAt(walking(lbbcgt_5),930). orientation(lbbcgt_6,144,930). coords(lbbcgt_6,97,82,930).  visible(lbbcgt_6,930). happensAt(walking(lbbcgt_6),930).
orientation(lbbcgt_5,158,931). coords(lbbcgt_5,204,190,931).  visible(lbbcgt_5,931). happensAt(walking(lbbcgt_5),931). orientation(lbbcgt_6,144,931). coords(lbbcgt_6,97,82,931).  visible(lbbcgt_6,931). happensAt(walking(lbbcgt_6),931).
orientation(lbbcgt_5,158,932). coords(lbbcgt_5,204,189,932).  visible(lbbcgt_5,932). happensAt(walking(lbbcgt_5),932). orientation(lbbcgt_6,144,932). coords(lbbcgt_6,97,83,932).  visible(lbbcgt_6,932). happensAt(walking(lbbcgt_6),932).
orientation(lbbcgt_5,158,933). coords(lbbcgt_5,203,187,933).  visible(lbbcgt_5,933). happensAt(walking(lbbcgt_5),933). orientation(lbbcgt_6,144,933). coords(lbbcgt_6,96,83,933).  visible(lbbcgt_6,933). happensAt(walking(lbbcgt_6),933).
#show holdsAt/2.