
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

	orientation(lbpugt_2,125,911). coords(lbpugt_2,227,120,911).  visible(lbpugt_2,911). happensAt(walking(lbpugt_2),911). orientation(lbpugt_4,135,911). coords(lbpugt_4,219,34,911).  visible(lbpugt_4,911). happensAt(inactive(lbpugt_4),911).
orientation(lbpugt_2,130,912). coords(lbpugt_2,225,119,912).  visible(lbpugt_2,912). happensAt(walking(lbpugt_2),912). orientation(lbpugt_4,135,912). coords(lbpugt_4,219,34,912).  visible(lbpugt_4,912). happensAt(inactive(lbpugt_4),912).
orientation(lbpugt_2,130,913). coords(lbpugt_2,225,118,913).  visible(lbpugt_2,913). happensAt(walking(lbpugt_2),913). orientation(lbpugt_4,135,913). coords(lbpugt_4,219,34,913).  visible(lbpugt_4,913). happensAt(inactive(lbpugt_4),913).
orientation(lbpugt_2,130,914). coords(lbpugt_2,224,117,914).  visible(lbpugt_2,914). happensAt(walking(lbpugt_2),914). orientation(lbpugt_4,135,914). coords(lbpugt_4,219,34,914).  visible(lbpugt_4,914). happensAt(inactive(lbpugt_4),914).
orientation(lbpugt_2,130,915). coords(lbpugt_2,223,117,915).  visible(lbpugt_2,915). happensAt(walking(lbpugt_2),915). orientation(lbpugt_4,135,915). coords(lbpugt_4,219,34,915).  visible(lbpugt_4,915). happensAt(inactive(lbpugt_4),915).
orientation(lbpugt_2,130,916). coords(lbpugt_2,222,117,916).  visible(lbpugt_2,916). happensAt(walking(lbpugt_2),916). orientation(lbpugt_4,135,916). coords(lbpugt_4,219,34,916).  visible(lbpugt_4,916). happensAt(inactive(lbpugt_4),916).
orientation(lbpugt_2,134,917). coords(lbpugt_2,221,117,917).  visible(lbpugt_2,917). happensAt(walking(lbpugt_2),917). orientation(lbpugt_4,135,917). coords(lbpugt_4,219,34,917).  visible(lbpugt_4,917). happensAt(inactive(lbpugt_4),917).
orientation(lbpugt_2,134,918). coords(lbpugt_2,220,116,918).  visible(lbpugt_2,918). happensAt(walking(lbpugt_2),918). orientation(lbpugt_4,135,918). coords(lbpugt_4,219,34,918).  visible(lbpugt_4,918). happensAt(inactive(lbpugt_4),918).
orientation(lbpugt_2,137,919). coords(lbpugt_2,219,115,919).  visible(lbpugt_2,919). happensAt(walking(lbpugt_2),919). orientation(lbpugt_4,135,919). coords(lbpugt_4,219,34,919).  visible(lbpugt_4,919). happensAt(inactive(lbpugt_4),919).
orientation(lbpugt_2,137,920). coords(lbpugt_2,217,114,920).  visible(lbpugt_2,920). happensAt(walking(lbpugt_2),920). orientation(lbpugt_4,135,920). coords(lbpugt_4,219,34,920).  visible(lbpugt_4,920). happensAt(inactive(lbpugt_4),920).
orientation(lbpugt_2,137,921). coords(lbpugt_2,215,112,921).  visible(lbpugt_2,921). happensAt(walking(lbpugt_2),921). orientation(lbpugt_4,135,921). coords(lbpugt_4,219,34,921).  visible(lbpugt_4,921). happensAt(inactive(lbpugt_4),921).
orientation(lbpugt_2,137,922). coords(lbpugt_2,213,110,922).  visible(lbpugt_2,922). happensAt(walking(lbpugt_2),922). orientation(lbpugt_4,135,922). coords(lbpugt_4,219,34,922).  visible(lbpugt_4,922). happensAt(inactive(lbpugt_4),922).
orientation(lbpugt_2,137,923). coords(lbpugt_2,211,108,923).  visible(lbpugt_2,923). happensAt(walking(lbpugt_2),923). orientation(lbpugt_4,135,923). coords(lbpugt_4,219,34,923).  visible(lbpugt_4,923). happensAt(inactive(lbpugt_4),923).
orientation(lbpugt_2,135,924). coords(lbpugt_2,209,107,924).  visible(lbpugt_2,924). happensAt(walking(lbpugt_2),924). orientation(lbpugt_4,135,924). coords(lbpugt_4,219,34,924).  visible(lbpugt_4,924). happensAt(inactive(lbpugt_4),924).
orientation(lbpugt_2,135,925). coords(lbpugt_2,207,106,925).  visible(lbpugt_2,925). happensAt(walking(lbpugt_2),925). orientation(lbpugt_4,135,925). coords(lbpugt_4,219,34,925).  visible(lbpugt_4,925). happensAt(inactive(lbpugt_4),925).
orientation(lbpugt_2,131,926). coords(lbpugt_2,205,105,926).  visible(lbpugt_2,926). happensAt(walking(lbpugt_2),926). orientation(lbpugt_4,135,926). coords(lbpugt_4,219,34,926).  visible(lbpugt_4,926). happensAt(inactive(lbpugt_4),926).
orientation(lbpugt_2,131,927). coords(lbpugt_2,205,105,927).  visible(lbpugt_2,927). happensAt(walking(lbpugt_2),927). orientation(lbpugt_4,135,927). coords(lbpugt_4,219,34,927).  visible(lbpugt_4,927). happensAt(inactive(lbpugt_4),927).
orientation(lbpugt_2,131,928). coords(lbpugt_2,204,104,928).  visible(lbpugt_2,928). happensAt(walking(lbpugt_2),928). orientation(lbpugt_4,135,928). coords(lbpugt_4,219,34,928).  visible(lbpugt_4,928). happensAt(inactive(lbpugt_4),928).
orientation(lbpugt_2,131,929). coords(lbpugt_2,204,104,929).  visible(lbpugt_2,929). happensAt(walking(lbpugt_2),929). orientation(lbpugt_4,135,929). coords(lbpugt_4,219,34,929).  visible(lbpugt_4,929). happensAt(inactive(lbpugt_4),929).
orientation(lbpugt_2,131,930). coords(lbpugt_2,204,103,930).  visible(lbpugt_2,930). happensAt(walking(lbpugt_2),930). orientation(lbpugt_4,135,930). coords(lbpugt_4,219,34,930).  visible(lbpugt_4,930). happensAt(inactive(lbpugt_4),930).
orientation(lbpugt_2,131,931). coords(lbpugt_2,203,103,931).  visible(lbpugt_2,931). happensAt(walking(lbpugt_2),931). orientation(lbpugt_4,135,931). coords(lbpugt_4,219,34,931).  visible(lbpugt_4,931). happensAt(inactive(lbpugt_4),931).
orientation(lbpugt_2,125,932). coords(lbpugt_2,203,102,932).  visible(lbpugt_2,932). happensAt(walking(lbpugt_2),932). orientation(lbpugt_4,135,932). coords(lbpugt_4,219,34,932).  visible(lbpugt_4,932). happensAt(inactive(lbpugt_4),932).
orientation(lbpugt_2,125,933). coords(lbpugt_2,203,101,933).  visible(lbpugt_2,933). happensAt(walking(lbpugt_2),933). orientation(lbpugt_4,135,933). coords(lbpugt_4,219,34,933).  visible(lbpugt_4,933). happensAt(inactive(lbpugt_4),933).
orientation(lbpugt_2,124,934). coords(lbpugt_2,202,99,934).  visible(lbpugt_2,934). happensAt(walking(lbpugt_2),934). orientation(lbpugt_4,135,934). coords(lbpugt_4,219,34,934).  visible(lbpugt_4,934). happensAt(inactive(lbpugt_4),934).
orientation(lbpugt_2,124,935). coords(lbpugt_2,201,97,935).  visible(lbpugt_2,935). happensAt(walking(lbpugt_2),935). orientation(lbpugt_4,135,935). coords(lbpugt_4,219,34,935).  visible(lbpugt_4,935). happensAt(inactive(lbpugt_4),935).
#show holdsAt/2.