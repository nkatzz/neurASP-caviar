
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

	orientation(wk2gt_0,12,12). coords(wk2gt_0,208,246,12).  visible(wk2gt_0,12). happensAt(walking(wk2gt_0),12). orientation(wk2gt_1,143,12). coords(wk2gt_1,103,65,12).  visible(wk2gt_1,12). happensAt(walking(wk2gt_1),12).
orientation(wk2gt_0,12,13). coords(wk2gt_0,208,248,13).  visible(wk2gt_0,13). happensAt(walking(wk2gt_0),13). orientation(wk2gt_1,143,13). coords(wk2gt_1,103,65,13).  visible(wk2gt_1,13). happensAt(walking(wk2gt_1),13).
orientation(wk2gt_0,12,14). coords(wk2gt_0,210,248,14).  visible(wk2gt_0,14). happensAt(walking(wk2gt_0),14). orientation(wk2gt_1,143,14). coords(wk2gt_1,104,65,14).  visible(wk2gt_1,14). happensAt(walking(wk2gt_1),14).
orientation(wk2gt_0,12,15). coords(wk2gt_0,210,251,15).  visible(wk2gt_0,15). happensAt(walking(wk2gt_0),15). orientation(wk2gt_1,143,15). coords(wk2gt_1,104,65,15).  visible(wk2gt_1,15). happensAt(walking(wk2gt_1),15).
orientation(wk2gt_0,12,16). coords(wk2gt_0,210,252,16).  visible(wk2gt_0,16). happensAt(walking(wk2gt_0),16). orientation(wk2gt_1,143,16). coords(wk2gt_1,104,65,16).  visible(wk2gt_1,16). happensAt(walking(wk2gt_1),16).
orientation(wk2gt_0,20,17). coords(wk2gt_0,210,253,17).  visible(wk2gt_0,17). happensAt(walking(wk2gt_0),17). orientation(wk2gt_1,143,17). coords(wk2gt_1,104,65,17).  visible(wk2gt_1,17). happensAt(walking(wk2gt_1),17).
orientation(wk2gt_0,20,18). coords(wk2gt_0,210,254,18).  visible(wk2gt_0,18). happensAt(walking(wk2gt_0),18). orientation(wk2gt_1,143,18). coords(wk2gt_1,105,64,18).  visible(wk2gt_1,18). happensAt(walking(wk2gt_1),18).
orientation(wk2gt_0,20,19). coords(wk2gt_0,210,256,19).  visible(wk2gt_0,19). happensAt(walking(wk2gt_0),19). orientation(wk2gt_1,143,19). coords(wk2gt_1,105,64,19).  visible(wk2gt_1,19). happensAt(walking(wk2gt_1),19).
orientation(wk2gt_0,31,20). coords(wk2gt_0,210,258,20).  visible(wk2gt_0,20). happensAt(walking(wk2gt_0),20). orientation(wk2gt_1,143,20). coords(wk2gt_1,105,64,20).  visible(wk2gt_1,20). happensAt(walking(wk2gt_1),20).
orientation(wk2gt_0,31,21). coords(wk2gt_0,211,261,21).  visible(wk2gt_0,21). happensAt(walking(wk2gt_0),21). orientation(wk2gt_1,143,21). coords(wk2gt_1,105,63,21).  visible(wk2gt_1,21). happensAt(walking(wk2gt_1),21).
orientation(wk2gt_0,31,22). coords(wk2gt_0,211,265,22).  visible(wk2gt_0,22). happensAt(walking(wk2gt_0),22). orientation(wk2gt_1,143,22). coords(wk2gt_1,105,62,22).  visible(wk2gt_1,22). happensAt(walking(wk2gt_1),22).
orientation(wk2gt_0,31,23). coords(wk2gt_0,212,268,23).  visible(wk2gt_0,23). happensAt(walking(wk2gt_0),23). orientation(wk2gt_1,143,23). coords(wk2gt_1,105,62,23).  visible(wk2gt_1,23). happensAt(walking(wk2gt_1),23).
orientation(wk2gt_0,31,24). coords(wk2gt_0,213,271,24).  visible(wk2gt_0,24). happensAt(walking(wk2gt_0),24). orientation(wk2gt_1,143,24). coords(wk2gt_1,105,62,24).  visible(wk2gt_1,24). happensAt(walking(wk2gt_1),24).
orientation(wk2gt_0,27,25). coords(wk2gt_0,213,273,25).  visible(wk2gt_0,25). happensAt(walking(wk2gt_0),25). orientation(wk2gt_1,143,25). coords(wk2gt_1,106,61,25).  visible(wk2gt_1,25). happensAt(walking(wk2gt_1),25).
orientation(wk2gt_0,27,26). coords(wk2gt_0,215,273,26).  visible(wk2gt_0,26). happensAt(walking(wk2gt_0),26). orientation(wk2gt_1,143,26). coords(wk2gt_1,106,60,26).  visible(wk2gt_1,26). happensAt(walking(wk2gt_1),26).
orientation(wk2gt_0,27,27). coords(wk2gt_0,216,273,27).  visible(wk2gt_0,27). happensAt(walking(wk2gt_0),27). orientation(wk2gt_1,143,27). coords(wk2gt_1,107,60,27).  visible(wk2gt_1,27). happensAt(walking(wk2gt_1),27).
orientation(wk2gt_0,32,28). coords(wk2gt_0,216,273,28).  visible(wk2gt_0,28). happensAt(walking(wk2gt_0),28). orientation(wk2gt_1,143,28). coords(wk2gt_1,108,60,28).  visible(wk2gt_1,28). happensAt(walking(wk2gt_1),28).
orientation(wk2gt_0,32,29). coords(wk2gt_0,217,273,29).  visible(wk2gt_0,29). happensAt(walking(wk2gt_0),29). orientation(wk2gt_1,143,29). coords(wk2gt_1,108,60,29).  visible(wk2gt_1,29). happensAt(walking(wk2gt_1),29).
orientation(wk2gt_0,39,30). coords(wk2gt_0,217,273,30).  visible(wk2gt_0,30). happensAt(walking(wk2gt_0),30). orientation(wk2gt_1,143,30). coords(wk2gt_1,108,60,30).  visible(wk2gt_1,30). happensAt(walking(wk2gt_1),30).
orientation(wk2gt_0,39,31). coords(wk2gt_0,218,274,31).  visible(wk2gt_0,31). happensAt(walking(wk2gt_0),31). orientation(wk2gt_1,143,31). coords(wk2gt_1,108,60,31).  visible(wk2gt_1,31). happensAt(walking(wk2gt_1),31).
orientation(wk2gt_0,39,32). coords(wk2gt_0,219,275,32).  visible(wk2gt_0,32). happensAt(walking(wk2gt_0),32). orientation(wk2gt_1,143,32). coords(wk2gt_1,108,60,32).  visible(wk2gt_1,32). happensAt(walking(wk2gt_1),32).
orientation(wk2gt_0,53,33). coords(wk2gt_0,222,277,33).  visible(wk2gt_0,33). happensAt(walking(wk2gt_0),33). orientation(wk2gt_1,143,33). coords(wk2gt_1,109,60,33).  visible(wk2gt_1,33). happensAt(walking(wk2gt_1),33).
orientation(wk2gt_0,53,34). coords(wk2gt_0,224,279,34).  visible(wk2gt_0,34). happensAt(walking(wk2gt_0),34). orientation(wk2gt_1,143,34). coords(wk2gt_1,109,59,34).  visible(wk2gt_1,34). happensAt(walking(wk2gt_1),34).
orientation(wk2gt_0,53,35). coords(wk2gt_0,224,280,35).  visible(wk2gt_0,35). happensAt(walking(wk2gt_0),35). orientation(wk2gt_1,143,35). coords(wk2gt_1,109,59,35).  visible(wk2gt_1,35). happensAt(walking(wk2gt_1),35).
orientation(wk2gt_0,70,36). coords(wk2gt_0,225,282,36).  visible(wk2gt_0,36). happensAt(walking(wk2gt_0),36). orientation(wk2gt_1,143,36). coords(wk2gt_1,109,58,36).  visible(wk2gt_1,36). happensAt(walking(wk2gt_1),36).
#show holdsAt/2.